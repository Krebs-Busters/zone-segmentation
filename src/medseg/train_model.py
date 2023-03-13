from typing import Optional, Tuple
from .data_loader import PicaiLoader
import numpy as np
import jax.numpy as jnp
import jax

import matplotlib.pyplot as plt
import optax
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
from tqdm import tqdm


def normalize(
    data: np.ndarray, mean: float = None, std: float = None
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



class UNet3D(nn.Module):
    """A 3d UNet, see https://arxiv.org/pdf/1505.04597.pdf and 
       https://www.var.ovgu.de/pub/2019_Meyer_ISBI_Zone_Segmentation.pdf
       for more information."""

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        # x: i.e. (8, 256, 256, 21)
        x1 = jnp.expand_dims(x ,-1).transpose([0, 3, 1, 2, 4])
        init_feat = 4 #8 # 16
        # TODO: switch to valid!
        x1 = nn.relu(nn.Conv(features=init_feat, kernel_size=(3,3,3), padding="SAME")(x1))
        x1 = nn.relu(nn.Conv(features=init_feat, kernel_size=(3,3,3), padding="SAME")(x1))
        x2 = nn.max_pool(x1, (2, 2), strides=(2,2))

        x2 = nn.relu(nn.Conv(features=init_feat*2, kernel_size=(3,3,3), padding="SAME")(x2))
        x2 = nn.relu(nn.Conv(features=init_feat*2, kernel_size=(3,3,3), padding="SAME")(x2))
        x3 = nn.max_pool(x2, (2, 2), strides=(2,2))
        
        x3 = nn.relu(nn.Conv(features=init_feat*4, kernel_size=(3,3,3), padding="SAME")(x3))
        x3 = nn.relu(nn.Conv(features=init_feat*4, kernel_size=(3,3,3), padding="SAME")(x3))
        x4 = nn.max_pool(x3, (2, 2), strides=(2,2))

        x4 = nn.relu(nn.Conv(features=init_feat*8, kernel_size=(3,3,3), padding="SAME")(x4))
        x4 = nn.relu(nn.Conv(features=init_feat*8, kernel_size=(3,3,3), padding="SAME")(x4))
        x5 = nn.max_pool(x4, (2, 2), strides=(2,2))

        x5 = nn.relu(nn.Conv(features=init_feat*16, kernel_size=(3,3,3), padding="SAME")(x5))
        x5 = nn.relu(nn.Conv(features=init_feat*16, kernel_size=(3,3,3), padding="SAME")(x5))

        x6 = nn.ConvTranspose(features=init_feat*8, kernel_size=(3,3,3), strides=(1,2,2))(x5)
        x6 = jnp.concatenate([x4, x6], axis=-1)
        x6 = nn.relu(nn.Conv(features=init_feat*8, kernel_size=(3,3,3), padding="SAME")(x6))
        x6 = nn.relu(nn.Conv(features=init_feat*8, kernel_size=(3,3,3), padding="SAME")(x6))

        x7 = nn.ConvTranspose(features=init_feat*4, kernel_size=(3,3,3), strides=(1,2,2))(x6)
        x7 = jnp.concatenate([x3, x7], axis=-1)
        x7 = nn.relu(nn.Conv(features=init_feat*4, kernel_size=(3,3,3), padding="SAME")(x7))
        x7 = nn.relu(nn.Conv(features=init_feat*4, kernel_size=(3,3,3), padding="SAME")(x7))

        x8 = nn.ConvTranspose(features=init_feat*2, kernel_size=(3,3,3), strides=(1,2,2))(x7)
        x8 = jnp.concatenate([x2, x8], axis=-1)
        x8 = nn.relu(nn.Conv(features=init_feat*2, kernel_size=(3,3,3), padding="SAME")(x8))
        x8 = nn.relu(nn.Conv(features=init_feat*2, kernel_size=(3,3,3), padding="SAME")(x8))

        x9 = nn.ConvTranspose(features=init_feat, kernel_size=(3,3,3), strides=(1,2,2))(x8)
        x9 = jnp.concatenate([x1, x9], axis=-1)
        x9 = nn.relu(nn.Conv(features=init_feat, kernel_size=(3,3,3), padding="SAME")(x9))
        x9 = nn.relu(nn.Conv(features=1, kernel_size=(3,3,3), padding="SAME")(x9))
        return x9.squeeze(-1).transpose(0, 2, 3, 1)



def train():
    
    input_shape = [256, 256, 21]
    data_set = PicaiLoader(workers=4, input_shape=input_shape)
    epochs = 20
    batch_size = 12
    opt = optax.adam(learning_rate=0.001)

    model = UNet3D()

    key = jax.random.PRNGKey(42)
    net_state = model.init(key, jnp.ones([batch_size] + input_shape))
    opt_state = opt.init(net_state)

    # @jax.jit
    def forward_step(
        variables: FrozenDict, img_batch: jnp.ndarray, label_batch: jnp.ndarray,
    ):
        """Do a forward step."""
        out = model.apply(variables, img_batch)
        ce_loss = jnp.mean(
            optax.softmax_cross_entropy(
                logits=out, labels=label_batch
            )
        )
        if jnp.max(label_batch) > 0.1:
            pass

        return ce_loss

    loss_grad_fn = jax.value_and_grad(forward_step)
    iterations_counter = 0
    epoch_counter = 0

    for e in range(epochs):
        for data_batch in data_set.get_epoch(batch_size):
            input_x = data_batch['images']
            labels_y = data_batch['annotation']

            input_x = normalize(input_x,
                                mean=np.array([3.4925714]),
                                std=np.array([31.330494]))
            cel, grads = loss_grad_fn(net_state, input_x, labels_y)
            updates, opt_state = opt.update(grads, opt_state, net_state)
            net_state = optax.apply_updates(net_state, updates)
            iterations_counter += 1
            print(f"epoch: {e}, iteration: {iterations_counter}, loss: {cel}")

    pass

if __name__ == '__main__':
    train()