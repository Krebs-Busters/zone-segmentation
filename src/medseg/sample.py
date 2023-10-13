"""Samples the test data, to asses segmentation quality."""

import pickle
from typing import Dict

import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from flax import linen as nn

from data_loader import Loader
from train import UNet3D, normalize

if __name__ == "__main__":
    mean = jnp.array([206.12558])
    std = jnp.array([164.74423])
    input_shape = [128, 128, 21]
    val_keys = ["ProstateX-0004", "ProstateX-0007"]
    data_set = Loader(input_shape=input_shape, val_keys=val_keys)
    model = UNet3D()

    val_data = data_set.get_val(True)
    with open("./weights/unet_499.pkl", "rb") as f:
        net_state = pickle.load(f)

    input_val = normalize(val_data["images"], mean=mean, std=std)
    val_out = model.apply(net_state, input_val)
    val_loss = jnp.mean(
        optax.softmax_cross_entropy(val_out, nn.one_hot(val_data["annotation"], 5))
    )
    print(f"Validation loss: {val_loss:2.6f}")

    def disp_result(
        sample: int, data: Dict[str, jnp.ndarray], out: jnp.ndarray, slice: int = 11
    ):
        """Plot the original image, network output and annotation."""
        plt.title("scan")
        plt.imshow(data["images"][sample, :, :, slice])
        plt.show()
        plt.title("network")
        plt.imshow(jnp.argmax(out[sample, :, :, slice], axis=-1), vmin=0, vmax=5)
        plt.show()
        plt.title("human expert")
        plt.imshow(data["annotation"][sample, :, :, slice], vmin=0, vmax=5)
        plt.show()

    disp_result(0, val_data, val_out)
    disp_result(1, val_data, val_out)

    test_data = data_set.get_test_set()
    input_test = normalize(test_data["images"], mean=mean, std=std)
    test_out = model.apply(net_state, input_test)
    test_loss = jnp.mean(
        optax.softmax_cross_entropy(test_out, nn.one_hot(test_data["annotation"], 5))
    )
    print(f"Test loss: {test_loss:2.6f}")

    for i in range(20):
        disp_result(i, test_data, test_out)

    pass
