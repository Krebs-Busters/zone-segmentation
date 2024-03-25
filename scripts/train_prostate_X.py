"""Train a Unet for medical image segmentation.

Methods as described in
https://arxiv.org/pdf/1505.04597.pdf and
https://www.var.ovgu.de/pub/2019_Meyer_ISBI_Zone_Segmentation.pdf.
"""

import argparse
import os
import pickle
from datetime import datetime
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import sklearn.metrics
import scipy.spatial.distance
from clu import metric_writers
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
from tqdm import tqdm

from src.medseg.data_loader import Loader
from src.medseg.networks import (
    normalize,
    softmax_focal_loss,
    sigmoid_focal_loss,
    UNet3D,
    save_network
    )


def _parse_args():
    """Parse cmd line args for training an image classifier."""
    parser = argparse.ArgumentParser(description="Train an image classifier")
    parser.add_argument(
        "--cost",
        choices=["ce", "sce", "sigfocal", "softfocal"],
        default="softfocal",
        help="the cost function",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="learning rate for optimizer (default: 1e-3)",
    )
    parser.add_argument(
        "--epochs", type=int, default=500, help="number of epochs (default: 500)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="number of epochs (default: 500)"
    )
    return parser.parse_args()


def train(args):
    """Train the Unet."""
    # Choose two scans for validation.
    val_keys = ["ProstateX-0004", "ProstateX-0007"]

    input_shape = [128, 128, 21]  # [168, 168, 32]
    data_set = Loader(input_shape=input_shape, val_keys=val_keys)
    epochs = args.epochs
    if args.cost == 'ce':
        cost = partial(optax.softmax_cross_entropy)
    elif args.cost == 'sce':
        cost = partial(optax.sigmoid_binary_cross_entropy)
    elif args.cost == 'sigfocal':
        cost = partial(sigmoid_focal_loss, gamma = 1.25)
    else:
        cost = partial(softmax_focal_loss, gamma=1.25)
    batch_size = 2
    # opt = optax.sgd(learning_rate=0.001, momentum=0.99)
    opt = optax.adam(learning_rate=0.001)
    load_new = True

    model = UNet3D()
    writer = metric_writers.create_default_writer(
        "./runs/" + f"{str(datetime.now())}_{cost.func.__name__}", asynchronous=False
    )

    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    net_state = model.init(key, jnp.ones([batch_size] + input_shape))
    opt_state = opt.init(net_state)

    @jax.jit
    def forward_step(
        variables: FrozenDict,
        img_batch: jnp.ndarray,
        label_batch: jnp.ndarray,
    ):
        """Do a forward step."""
        labels = nn.one_hot(label_batch, 5)
        out = model.apply(variables, img_batch)
        loss = jnp.mean(cost(out, labels))
        return loss

    loss_grad_fn = jax.value_and_grad(forward_step)
    iterations_counter = 0

    if load_new:
        epoch_batches = list(data_set.get_epoch(batch_size))

        if not os.path.exists("./data/pickled/"):
            os.makedirs("./data/pickled/")

        with open("./data/pickled/batch_dump.pkl", "wb") as file:
            pickle.dump(epoch_batches, file)
    else:
        with open("./data/pickled/batch_dump.pkl", "rb") as file:
            epoch_batches = pickle.load(file)

    val_data = data_set.get_val()
    mean = jnp.array([248.29199])
    std = jnp.array([159.64618])
    val_loss_list = []
    train_loss_list = []
    val_loss = 0

    bar = tqdm(range(epochs))
    for e in bar:
        np.random.shuffle(epoch_batches)
        # epoch_batches_pre = flax.jax_utils.prefetch_to_device(
        #     iter(epoch_batches), 2, [jax.devices()[0]]
        # )
        for data_batch in iter(epoch_batches):
            input_x = data_batch["images"]
            labels_y = data_batch["annotation"]
            input_x = normalize(input_x, mean=mean, std=std)
            cel, grads = loss_grad_fn(net_state, input_x, labels_y)
            updates, opt_state = opt.update(grads, opt_state, net_state)
            net_state = optax.apply_updates(net_state, updates)
            iterations_counter += 1
            bar.set_description(f"epoch: {e}, loss: {cel:2.6f}, val-loss: {val_loss:2.6f}")
            train_loss_list.append((iterations_counter, cel))
            writer.write_scalars(iterations_counter, {"training_loss": cel})

        # epoch ended validate
        input_val = normalize(val_data["images"], mean=mean, std=std)
        val_out = model.apply(net_state, input_val)
        val_netseg = jnp.argmax(val_out, -1)
    
        val_loss = jnp.mean(
            optax.softmax_cross_entropy(val_out, nn.one_hot(val_data["annotation"], 5))
        )
        val_iou = sklearn.metrics.jaccard_score(val_data["annotation"].flatten(),
                                                val_netseg.flatten(), average='weighted')
        val_dice = scipy.spatial.distance.dice(val_data["annotation"].flatten(),
                                               val_netseg.flatten())
        val_mean_dist = jnp.mean(jnp.abs(val_data["annotation"] - val_netseg))
        val_loss_list.append((iterations_counter, val_loss))
        writer.write_scalars(iterations_counter, 
                             {"validation_ce_loss": val_loss,
                              "validation_iou": val_iou,
                              "val_dice": val_dice,
                              "val_mad": val_mean_dist})
        for i in range(len(val_keys)):
            writer.write_images(
                iterations_counter,
                {
                    f"{i}_val_network_segmentation": jnp.expand_dims(
                        jnp.argmax(val_out[i, :, :, 12], -1), (0, -1)
                    )
                    / 5.0
                },
            )
            writer.write_images(
                iterations_counter,
                {
                    f"{i}_val_true_segmentation": jnp.expand_dims(
                        val_data["annotation"][i, :, :, 12] / 5.0, (0, -1)
                    )
                },
            )
        if e % 20 == 0:
            save_network(net_state, e, )

    tll = np.stack(train_loss_list, -1)
    vll = np.stack(val_loss_list, -1)
    #plt.semilogy(tll[0], tll[1])
    #plt.semilogy(vll[0], vll[1])
    #plt.show()
    #
    #plt.imshow(val_data["annotation"][0, :, :, 12])
    #plt.show()
    #plt.imshow(jnp.argmax(val_out[0, :, :, 12], axis=-1))
    #plt.show()

    if not os.path.exists("./weights/"):
        os.makedirs("./weights/")

    print("Training done.")
    save_network(net_state, e, cost.func.__name__)


if __name__ == "__main__":
    args = _parse_args()
    train(args)
