from data_loader import PicaiLoader
import jax.numpy as jnp

import matplotlib.pyplot as plt
from flax import linen as nn


class UNet3D(nn.module):
    """A 3d UNet."""


    @nn.compact
    def __cal__(self, x: jnp.ndarray):
        # x: i.e. (8, 256, 256, 21)
        x1 = nn.Conv(features=16, kernel_size=(3,3,3))
        x1 = nn.Conv(features=32, kernel_size=(3,3,3))
        x2 = nn.max_pool(x1, (2,2))
        return None


def train():
    
    data_set = PicaiLoader()
    epochs = 20

    model = None

    batch = data_set.get_batch(6)
    batch_x = batch['images']['t2w']
    batch_y = batch['annotation']
    pass



if __name__ == '__main__':
    train()