"""Samples the test data, to asses segmentation quality."""
import pickle
import argparse

import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from src.medseg.data_loader import Loader
from flax import linen as nn
from src.medseg.networks import UNet3D, normalize, dice_similarity_coef
from src.medseg.util import disp_result


def _parse_args():
    """Parse cmd line args for training an image classifier."""
    parser = argparse.ArgumentParser(description="Train an image classifier")
    parser.add_argument(
        "path_to_weights",
        type=str,
        help="where to look for pickled network weights.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    mean = jnp.array([248.29199])
    std = jnp.array([159.64618])
    input_shape = [128, 128, 21]
    val_keys = ["ProstateX-0004", "ProstateX-0007"]
    data_set = Loader(input_shape=input_shape, val_keys=val_keys)
    model = UNet3D()

    val_data = data_set.get_val(True)
    with open(args.path_to_weights, "rb") as f:
        net_state = pickle.load(f)

    input_val = normalize(val_data["images"], mean=mean, std=std)
    val_out = model.apply(net_state, input_val)
    val_loss = jnp.mean(
        optax.softmax_cross_entropy(val_out, nn.one_hot(val_data["annotation"], 5))
    )
    print(f"Validation loss: {val_loss:2.6f}")

    disp_result(val_data['images'][0], jnp.expand_dims(jnp.argmax(val_out[0], -1), 0), id=0)
    disp_result(val_data['images'][1], jnp.expand_dims(jnp.argmax(val_out[1], -1), 0), id=1)

    test_data = data_set.get_test_set()
    input_test = normalize(test_data["images"], mean=mean, std=std)
    test_out = model.apply(net_state, input_test)
    test_loss = jnp.mean(
        optax.softmax_cross_entropy(test_out, nn.one_hot(test_data["annotation"], 5))
    )
    print(f"Test loss: {test_loss:2.6f}")

    y_pred = jnp.argmax(test_out, -1)
    test_dice = dice_similarity_coef(test_data["annotation"], y_pred)
    print(f"Overall - Test DICE-Score: {test_dice:2.2f}")
    
    pz_sections_annos = (test_data["annotation"] == 1).astype(jnp.float32)
    pz_pred = (y_pred == 1).astype(jnp.float32)
    pz_dice = dice_similarity_coef(pz_sections_annos, pz_pred)
    del pz_sections_annos
    del pz_pred

    tz_sections_annos = (test_data["annotation"] == 2).astype(jnp.float32)
    tz_pred = (y_pred == 2).astype(jnp.float32)
    tz_dice = dice_similarity_coef(tz_sections_annos, tz_pred)
    del tz_sections_annos
    del tz_pred

    print(f"PZ-Dice: {pz_dice}")
    print(f"TZ-Dice: {tz_dice}")
    
    test_pred = jnp.argmax(test_out, -1)

    for i in range(20):
         disp_result(test_data['images'][i], jnp.expand_dims(test_pred[i], 0), str(i))


