"""The module sets up our networks and cost functions."""

import pickle
from typing import Optional

import chex
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
from optax.losses import sigmoid_binary_cross_entropy


def softmax_focal_loss(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    alpha: jnp.ndarray = None,
    gamma: float = 2,
) -> jnp.ndarray:
    """Compute a softmax focal loss."""
    if not alpha:
        alpha = jnp.ones_like(logits)

    chex.assert_type([logits], float)
    # see also the original sigmoid implementation at:
    # https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/focal_loss.py
    chex.assert_type([logits], float)
    focus = jnp.power(-jax.nn.softmax(logits, axis=-1) + 1.0, gamma)
    loss = -labels * alpha * focus * jax.nn.log_softmax(logits, axis=-1)
    return jnp.sum(loss, axis=-1)


def sigmoid_focal_loss(
    logits: chex.Array,
    labels: chex.Array,
    alpha: Optional[float] = None,
    gamma: float = 2.0,
) -> chex.Array:
    """Sigmoid focal loss.

    The focal loss is a re-weighted cross entropy for unbalanced problems.
    Use this loss function if classes are not mutually exclusive.
    See `sigmoid_binary_cross_entropy` for more information.

    References:
      Lin et al. 2018. https://arxiv.org/pdf/1708.02002.pdf

    Args:
      logits: Array of floats. The predictions for each example.
        The predictions for each example.
      labels: Array of floats. Labels and logits must have
        the same shape. The label array must contain the binary
        classification labels for each element in the data set
        (0 for the out-of-class and 1 for in-class).
      alpha: (optional) Weighting factor in range (0,1) to balance
        positive vs negative examples. Default None (no weighting).
      gamma: Exponent of the modulating factor (1 - p_t).
        Balances easy vs hard examples.

    Returns:
      A loss value array with a shape identical to the logits and target
      arrays.
    """
    alpha = -1 if alpha is None else alpha

    chex.assert_type([logits], float)
    labels = labels.astype(logits.dtype)
    # see also the original paper's implementation at:
    # https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/focal_loss.py
    p = jax.nn.sigmoid(logits)
    ce_loss = sigmoid_binary_cross_entropy(logits, labels)
    p_t = p * labels + (1 - p) * (1 - labels)
    loss = ce_loss * ((1 - p_t) ** gamma)

    weighted = (
        lambda loss_arg: (alpha * labels + (1 - alpha) * (1 - labels)) * loss_arg
    )  # noqa: E731
    not_weighted = lambda loss_arg: loss_arg  # noqa: E731

    loss = jax.lax.cond(alpha >= 0, weighted, not_weighted, loss)
    return loss


def dice_similarity_coef(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    """Compute the dice similarity coefficient.

    Args:
        y_true (jnp.ndarray): Desired output labels.
        y_pred (jnp.ndarray): Output label estimation.

    Returns:
        jnp.ndarray: Similarity as float in [0, 1].
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = jnp.sum(y_true_f * y_pred_f)
    return (2.0 * intersection) / (jnp.sum(y_true_f) + jnp.sum(y_pred_f))


def dice(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    """Compute the dice dissimilariy like scipy does.

    Args:
        y_true (jnp.ndarray): Desired output labels.
        y_pred (jnp.ndarray): Output label estimation.

    Returns:
        jnp.ndarray: Dissimilarity as float in [0, 1].
    """
    return 1.0 - dice_similarity_coef(y_true, y_pred)


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


def save_network(net_state: FrozenDict, epoch: int, info: str = "") -> None:
    """Save the network weights.

    Args:
        net_state (FrozenDict): The current network state.
        epoch (int): The number of epochs the network saw.
        info (str): A network identifier.
    """
    if info:
        name = f"./weights/unet_{info}_epoch_{epoch}.pkl"
    else:
        name = f"./weights/unet_epoch_{epoch}.pkl"

    with open(name, "wb") as f:
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
