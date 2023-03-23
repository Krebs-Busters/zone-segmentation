from typing import Optional, Tuple, List
from .data_loader import PicaiLoader
import numpy as np
import jax.numpy as jnp
import jax

import matplotlib.pyplot as plt
import optax
import flax
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
from tqdm import tqdm
import pickle


@jax.jit
def normalize(
    data: jnp.ndarray, mean: jnp.ndarray = None, std: jnp.ndarray = None
) -> np.ndarray:
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
        np.array, float, float: Normalized data, mean and std.
    """
    return (data - mean) / std


@jax.jit
def pad_odd(input_x: jnp.ndarray) -> jnp.ndarray:
    # dont pad the batch axis.
    pad_list = [(0, 0, 0)]
    for axis_shape in input_x.shape[1:-1]:
        if axis_shape % 2 != 0:
            pad_list.append((0, 1, 0))
        else:
            pad_list.append((0, 0, 0))
    # dont pad the features
    pad_list.append((0, 0, 0))
    return jax.lax.pad(input_x, 0., pad_list)



class UNet3D(nn.Module):
    """A 3d UNet, see https://arxiv.org/pdf/1505.04597.pdf and 
       https://www.var.ovgu.de/pub/2019_Meyer_ISBI_Zone_Segmentation.pdf
       for more information."""

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        # x: i.e. (8, 256, 256, 21) transform to NDHWC
        x1 = jnp.expand_dims(x ,-1).transpose([0, 3, 1, 2, 4])
        init_feat = 8 # 16
        out_neurons = 2
        x1 = nn.relu(nn.Conv(features=init_feat, kernel_size=(3,3,3), padding="SAME")(x1))
        x1 = nn.relu(nn.Conv(features=init_feat, kernel_size=(3,3,3), padding="SAME")(x1))
        x1 = pad_odd(x1)
        x2 = nn.max_pool(x1, (2, 2), strides=(2,2))

        x2 = nn.relu(nn.Conv(features=init_feat*2, kernel_size=(3,3,3), padding="SAME")(x2))
        x2 = nn.relu(nn.Conv(features=init_feat*2, kernel_size=(3,3,3), padding="SAME")(x2))
        x2 = pad_odd(x2)
        x3 = nn.max_pool(x2, (2, 2), strides=(2,2))
        
        x3 = nn.relu(nn.Conv(features=init_feat*4, kernel_size=(3,3,3), padding="SAME")(x3))
        x3 = nn.relu(nn.Conv(features=init_feat*4, kernel_size=(3,3,3), padding="SAME")(x3))
        x3 = pad_odd(x3)
        x4 = nn.max_pool(x3, (2, 2), strides=(2,2))

        x4 = nn.relu(nn.Conv(features=init_feat*8, kernel_size=(3,3,3), padding="SAME")(x4))
        x4 = nn.relu(nn.Conv(features=init_feat*8, kernel_size=(3,3,3), padding="SAME")(x4))
        x4 = pad_odd(x4)
        x5 = nn.max_pool(x4, (2, 2), strides=(2,2))

        x5 = nn.relu(nn.Conv(features=init_feat*16, kernel_size=(3,3,3), padding="SAME")(x5))
        x5 = nn.relu(nn.Conv(features=init_feat*16, kernel_size=(3,3,3), padding="SAME")(x5))

        x6 = nn.ConvTranspose(features=init_feat*8, kernel_size=(3,3,3), strides=(1,2,2))(x5)
        x6 = x6[:, :x4.shape[1], :x4.shape[2], :x4.shape[3], :]
        x6 = jnp.concatenate([x4, x6], axis=-1)
        x6 = nn.relu(nn.Conv(features=init_feat*8, kernel_size=(3,3,3), padding="SAME")(x6))
        x6 = nn.relu(nn.Conv(features=init_feat*8, kernel_size=(3,3,3), padding="SAME")(x6))

        x7 = nn.ConvTranspose(features=init_feat*4, kernel_size=(3,3,3), strides=(1,2,2))(x6)
        x7 = x7[:, :x3.shape[1], :x3.shape[2], :x3.shape[3], :]
        x7 = jnp.concatenate([x3, x7], axis=-1)
        x7 = nn.relu(nn.Conv(features=init_feat*4, kernel_size=(3,3,3), padding="SAME")(x7))
        x7 = nn.relu(nn.Conv(features=init_feat*4, kernel_size=(3,3,3), padding="SAME")(x7))

        x8 = nn.ConvTranspose(features=init_feat*2, kernel_size=(3,3,3), strides=(1,2,2))(x7)
        x8 = x8[:, :x2.shape[1], :x2.shape[2], :x2.shape[3], :]        
        x8 = jnp.concatenate([x2, x8], axis=-1)
        x8 = nn.relu(nn.Conv(features=init_feat*2, kernel_size=(3,3,3), padding="SAME")(x8))
        x8 = nn.relu(nn.Conv(features=init_feat*2, kernel_size=(3,3,3), padding="SAME")(x8))

        x9 = nn.ConvTranspose(features=init_feat, kernel_size=(3,3,3), strides=(1,2,2))(x8)
        x9 = x9[:, :x1.shape[1], :x1.shape[2], :x1.shape[3], :]
        x9 = jnp.concatenate([x1, x9], axis=-1)
        x9 = nn.relu(nn.Conv(features=init_feat, kernel_size=(3,3,3), padding="SAME")(x9))
        x9 = nn.relu(nn.Conv(features=out_neurons, kernel_size=(3,3,3), padding="SAME")(x9))
        return x9.transpose(0, 2, 3, 1, 4)[:, :x.shape[1], :x.shape[2], :x.shape[3], :]


def train():
    val_keys = ['10085_1000085', '10730_1000746', '10525_1000535']

    input_shape = [256, 256, 21]
    data_set = PicaiLoader(input_shape=input_shape, val_keys=val_keys)
    epochs = 50
    batch_size = 1
    opt = optax.sgd(learning_rate=0.001, momentum=0.99)
    load_new = False

    model = UNet3D()

    np.random.seed(42)
    key = jax.random.PRNGKey(42)
    net_state = model.init(key, jnp.ones([batch_size] + input_shape))
    opt_state = opt.init(net_state)

    @jax.jit
    def forward_step(
        variables: FrozenDict, img_batch: jnp.ndarray, label_batch: jnp.ndarray,
    ):
        """Do a forward step."""
        out = model.apply(variables, img_batch)
        ce_loss = jnp.mean(
            optax.softmax_cross_entropy(
                logits=out, labels=nn.one_hot(label_batch, 2)
            )
        )
        return ce_loss

    loss_grad_fn = jax.value_and_grad(forward_step)
    iterations_counter = 0

    if load_new:
        epoch_batches = list(data_set.get_epoch(batch_size))
        with open('./data/pickled/batch_dump.pkl', 'wb') as file:
            pickle.dump(epoch_batches, file)
    else:
        with open('./data/pickled/batch_dump.pkl', 'rb') as file:
            epoch_batches = pickle.load(file)



    val_data = data_set.get_val(batch_size=batch_size)
    mean = jnp.array([206.12558])
    std = jnp.array([164.74423])
    val_loss_list = []
    train_loss_list = []

    for e in range(epochs):
        np.random.shuffle(epoch_batches)
        epoch_batches_pre = flax.jax_utils.prefetch_to_device(iter(epoch_batches), 2, [jax.devices()[0]])
        bar = tqdm(epoch_batches_pre, desc="Training UNET", total=len(epoch_batches))
        for data_batch in bar:
            input_x = data_batch['images']
            labels_y = data_batch['annotation']
            input_x = normalize(input_x,
                                mean=mean,
                                std=std)
            cel, grads = loss_grad_fn(net_state, input_x, labels_y)
            updates, opt_state = opt.update(grads, opt_state, net_state)
            net_state = optax.apply_updates(net_state, updates)
            iterations_counter += 1
            bar.set_description(f"epoch: {e}, loss: {cel:2.6f}")
            train_loss_list.append((iterations_counter, cel))

        input_val = normalize(val_data[0]['images'], mean=mean, std=std)
        val_out = model.apply(net_state, input_val)
        val_loss = jnp.mean(optax.softmax_cross_entropy(
            val_out, nn.one_hot(val_data[0]['annotation'], 2)))
        print(f"Validation loss {val_loss:2.6f}")
        val_loss_list.append((iterations_counter, val_loss))
        # plt.imshow(jnp.max(nn.softmax(val_out)[0, ..., 1], -1)); plt.show()

    tll = np.stack(train_loss_list, -1)
    vll = np.stack(val_loss_list, -1)
    plt.semilogy(tll[0], tll[1])
    plt.semilogy(vll[0], vll[1])
    plt.show()

    with open('./weights/picaiunet.pkl', 'wb') as f:
        pickle.dump(net_state, f)

    print("Training done.")

if __name__ == '__main__':
    train()