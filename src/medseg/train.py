"""Train a Unet for medical image segmentation.

Methods as described in
https://arxiv.org/pdf/1505.04597.pdf and
https://www.var.ovgu.de/pub/2019_Meyer_ISBI_Zone_Segmentation.pdf.
"""

import os
import pickle
from datetime import datetime

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from clu import metric_writers
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
from tqdm import tqdm

from data_loader import Loader
from util import softmax_focal_loss


@jax.jit
def normalize(
    data: jnp.ndarray, mean: jnp.ndarray = None, std: jnp.ndarray = None
) -> jnp.ndarray:
    """Normalize the input array.

    After normalization the input
    distribution should be approximately standard normal.

    Args:
        data (np.array): The input array.
        mean (float): Data mean, re-computed if None.
            Defaults to None.
        std (float): Data standard deviation,
            re-computed if None. Defaults to None.

    Returns:
        jnp.array, float, float: Normalized data, mean and std.
    """
    return (data - mean) / std


def save_network(net_state: FrozenDict, epoch: int) -> None:
    """Save the network weights.

    Args:
        net_state (FrozenDict): The current network state.
        epoch (int): The number of epochs the network saw.
    """
    with open(f"./weights/unet_softmaxfl_{epoch}.pkl", "wb") as f:
        pickle.dump(net_state, f)


@jax.jit
def pad_odd(input_x: jnp.ndarray) -> jnp.ndarray:
    """Padd axes of input array to even length.

    Args:
        input_x (jnp.ndarray): The input array
            with axes of potential uneven length.

    Returns:
        jnp.ndarray: Output array with even dimension
            numbers.
    """
    # dont pad the batch axis.
    pad_list = [(0, 0, 0)]
    for axis_shape in input_x.shape[1:-1]:
        if axis_shape % 2 != 0:
            pad_list.append((0, 1, 0))
        else:
            pad_list.append((0, 0, 0))
    # dont pad the features
    pad_list.append((0, 0, 0))
    return jax.lax.pad(input_x, 0.0, pad_list)


class UNet3D(nn.Module):
    """A 3d UNet.

    See https://arxiv.org/pdf/1505.04597.pdf and
    https://www.var.ovgu.de/pub/2019_Meyer_ISBI_Zone_Segmentation.pdf
    for more information.
    """

    transpose_conv = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Call the unet.

        Args:
            x (jnp.ndarray): The input t2w scan.

        Returns:
            jnp.ndarray: The segmentation logits.
        """
        # x: i.e. (8, 256, 256, 21) transform to NDHWC
        x1 = jnp.expand_dims(x, -1).transpose([0, 3, 1, 2, 4])
        init_feat = 16
        out_neurons = 5
        x1 = nn.relu(
            nn.Conv(features=init_feat, kernel_size=(3, 3, 3), padding="SAME")(x1)
        )
        x1 = nn.relu(
            nn.Conv(features=init_feat, kernel_size=(3, 3, 3), padding="SAME")(x1)
        )
        x1 = pad_odd(x1)
        x2 = nn.max_pool(x1, (1, 2, 2), strides=(1, 2, 2))

        x2 = nn.relu(
            nn.Conv(features=init_feat * 2, kernel_size=(3, 3, 3), padding="SAME")(x2)
        )
        x2 = nn.relu(
            nn.Conv(features=init_feat * 2, kernel_size=(3, 3, 3), padding="SAME")(x2)
        )
        x2 = pad_odd(x2)
        x3 = nn.max_pool(x2, (1, 2, 2), strides=(1, 2, 2))

        x3 = nn.relu(
            nn.Conv(features=init_feat * 4, kernel_size=(3, 3, 3), padding="SAME")(x3)
        )
        x3 = nn.relu(
            nn.Conv(features=init_feat * 4, kernel_size=(3, 3, 3), padding="SAME")(x3)
        )
        x3 = pad_odd(x3)
        x4 = nn.max_pool(x3, (1, 2, 2), strides=(1, 2, 2))

        x4 = nn.relu(
            nn.Conv(features=init_feat * 8, kernel_size=(3, 3, 3), padding="SAME")(x4)
        )
        x4 = nn.relu(
            nn.Conv(features=init_feat * 8, kernel_size=(3, 3, 3), padding="SAME")(x4)
        )
        x4 = pad_odd(x4)
        x5 = nn.max_pool(x4, (1, 2, 2), strides=(1, 2, 2))

        x5 = nn.relu(
            nn.Conv(features=init_feat * 16, kernel_size=(3, 3, 3), padding="SAME")(x5)
        )
        x5 = nn.relu(
            nn.Conv(features=init_feat * 16, kernel_size=(3, 3, 3), padding="SAME")(x5)
        )

        def up_block(x_in):
            if self.transpose_conv:
                x_out = nn.ConvTranspose(
                    features=init_feat * 8, kernel_size=(3, 3, 3), strides=(1, 2, 2)
                )(x_in)
            else:
                b, d, h, w, c = x_in.shape
                x_out = jax.image.resize(x_in, (b, d, h * 2, w * 2, c), "nearest")
            return x_out

        x6 = up_block(x5)
        x6 = x6[:, : x4.shape[1], : x4.shape[2], : x4.shape[3], :]
        x6 = jnp.concatenate([x4, x6], axis=-1)
        x6 = nn.relu(
            nn.Conv(features=init_feat * 8, kernel_size=(3, 3, 3), padding="SAME")(x6)
        )
        x6 = nn.relu(
            nn.Conv(features=init_feat * 8, kernel_size=(3, 3, 3), padding="SAME")(x6)
        )

        x7 = up_block(x6)
        x7 = x7[:, : x3.shape[1], : x3.shape[2], : x3.shape[3], :]
        x7 = jnp.concatenate([x3, x7], axis=-1)
        x7 = nn.relu(
            nn.Conv(features=init_feat * 4, kernel_size=(3, 3, 3), padding="SAME")(x7)
        )
        x7 = nn.relu(
            nn.Conv(features=init_feat * 4, kernel_size=(3, 3, 3), padding="SAME")(x7)
        )

        x8 = up_block(x7)
        x8 = x8[:, : x2.shape[1], : x2.shape[2], : x2.shape[3], :]
        x8 = jnp.concatenate([x2, x8], axis=-1)
        x8 = nn.relu(
            nn.Conv(features=init_feat * 2, kernel_size=(3, 3, 3), padding="SAME")(x8)
        )
        x8 = nn.relu(
            nn.Conv(features=init_feat * 2, kernel_size=(3, 3, 3), padding="SAME")(x8)
        )

        x9 = up_block(x8)
        x9 = x9[:, : x1.shape[1], : x1.shape[2], : x1.shape[3], :]
        x9 = jnp.concatenate([x1, x9], axis=-1)
        x9 = nn.relu(
            nn.Conv(features=init_feat, kernel_size=(3, 3, 3), padding="SAME")(x9)
        )
        x9 = nn.relu(
            nn.Conv(features=out_neurons, kernel_size=(3, 3, 3), padding="SAME")(x9)
        )
        return x9.transpose(0, 2, 3, 1, 4)[
            :, : x.shape[1], : x.shape[2], : x.shape[3], :
        ]


def train():
    """Train the Unet."""
    # Choose two scans for validation.
    val_keys = ["ProstateX-0004", "ProstateX-0007"]

    input_shape = [128, 128, 21]  # [168, 168, 32]
    data_set = Loader(input_shape=input_shape, val_keys=val_keys)
    epochs = 1000
    batch_size = 2
    # opt = optax.sgd(learning_rate=0.001, momentum=0.99)
    opt = optax.adam(learning_rate=0.001)
    load_new = True

    model = UNet3D()
    writer = metric_writers.create_default_writer(
        "./runs/" + str(datetime.now()), asynchronous=False
    )

    np.random.seed(42)
    key = jax.random.PRNGKey(42)
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
        # loss = jnp.mean(
        #      optax.softmax_cross_entropy(logits=out, labels=labels)
        # )
        # loss = jnp.mean(sigmoid_focal_loss(
        #    logits=out, labels=labels, alpha=-1, gamma=2))
        loss = jnp.mean(softmax_focal_loss(out, labels, np.ones([out.shape[-1]])))
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
    mean = jnp.array([206.12558])
    std = jnp.array([164.74423])
    val_loss_list = []
    train_loss_list = []

    for e in range(epochs):
        np.random.shuffle(epoch_batches)
        # epoch_batches_pre = flax.jax_utils.prefetch_to_device(
        #     iter(epoch_batches), 2, [jax.devices()[0]]
        # )
        epoch_batches_pre = iter(epoch_batches)
        bar = tqdm(epoch_batches_pre, desc="Training UNET", total=len(epoch_batches))
        for data_batch in bar:
            input_x = data_batch["images"]
            labels_y = data_batch["annotation"]
            input_x = normalize(input_x, mean=mean, std=std)
            cel, grads = loss_grad_fn(net_state, input_x, labels_y)
            updates, opt_state = opt.update(grads, opt_state, net_state)
            net_state = optax.apply_updates(net_state, updates)
            iterations_counter += 1
            bar.set_description(f"epoch: {e}, loss: {cel:2.6f}")
            train_loss_list.append((iterations_counter, cel))
            writer.write_scalars(iterations_counter, {"training_loss": cel})
        input_val = normalize(val_data["images"], mean=mean, std=std)
        val_out = model.apply(net_state, input_val)
        val_loss = jnp.mean(
            optax.softmax_cross_entropy(val_out, nn.one_hot(val_data["annotation"], 5))
        )
        # todo: measure focal validation loss.
        print(f"Validation loss: {val_loss:2.6f}")
        val_loss_list.append((iterations_counter, val_loss))
        writer.write_scalars(iterations_counter, {"validation_loss": val_loss})
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
            save_network(net_state, e)

    tll = np.stack(train_loss_list, -1)
    vll = np.stack(val_loss_list, -1)
    plt.semilogy(tll[0], tll[1])
    plt.semilogy(vll[0], vll[1])
    plt.show()

    plt.imshow(val_data["annotation"][0, :, :, 12])
    plt.show()
    plt.imshow(jnp.argmax(val_out[0, :, :, 12], axis=-1))
    plt.show()

    if not os.path.exists("./weights/"):
        os.makedirs("./weights/")

    print("Training done.")
    save_network(net_state, e)


if __name__ == "__main__":
    train()
