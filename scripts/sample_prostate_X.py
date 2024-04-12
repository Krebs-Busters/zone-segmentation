"""Samples the test data, to asses segmentation quality."""
import pickle
import argparse

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import sklearn.metrics
from flax import linen as nn

from src.medseg.data_loader import Loader
from src.medseg.networks import UNet3D, normalize, dice_similarity_coef
from src.medseg.util import disp_result

jax.config.update('jax_platform_name', 'cpu')

def _parse_args():
    """Parse cmd line args for training an image classifier."""
    parser = argparse.ArgumentParser(description="Train an image classifier")
    parser.add_argument(
        "--path_to_weights",
        type=str,
        help="where to look for pickled network weights.",
        default="./weights/done/unet_epoch_2024-04-11_16:31:55.652071_softmax_focal_loss_g_1.5_499.pkl"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    print(args)
    mean = jnp.array([248.29199])
    std = jnp.array([159.64618])
    input_shape = [168, 168, 32]
    val_keys = ["ProstateX-0004"]
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

    # disp_result(val_data['images'][0], jnp.expand_dims(jnp.argmax(val_out[0], -1), 0), id=0)
    # disp_result(val_data['images'][1], jnp.expand_dims(jnp.argmax(val_out[1], -1), 0), id=1)

    test_data = data_set.get_test_set()
    input_test = normalize(test_data["images"], mean=mean, std=std)
    test_out = model.apply(net_state, input_test)
    test_loss = jnp.mean(
        optax.softmax_cross_entropy(test_out, nn.one_hot(test_data["annotation"], 5))
    )
    print(f"Test loss: {test_loss:2.6f}")

    y_pred = jnp.argmax(test_out, -1)
    
    dice_scores = []
    jaccard_scores = []

    zones = ['background', 'peripheral zone (pz)', 'transition zone (tz)', '*', 'AFS']
    print("zone order:", zones)

    for class_no in range(5):
        mask = test_data["annotation"] == class_no
        mask_pred = (y_pred[mask] == class_no).astype(int)
        mask_test_data = (test_data["annotation"][mask] == class_no).astype(int)

        dice_scores.append(float(dice_similarity_coef(mask_test_data, mask_pred)))
        jaccard_scores.append(sklearn.metrics.jaccard_score(
            mask_test_data, mask_pred,  average='binary', zero_division='warn'))

    print(f"per_class_dice_scores: {dice_scores}")
    print(f"per_class_IOU-scores: {jaccard_scores}")

    # test IOU
    test_iou = sklearn.metrics.jaccard_score(test_data["annotation"].flatten(),
                                             y_pred.flatten(), average='weighted')
    print(f"overall-test_iou: {test_iou}")


    test_pred = jnp.argmax(test_out, -1)

    # for i in range(20):
    #      disp_result(test_data['images'][i], jnp.expand_dims(test_pred[i], 0), str(i))
