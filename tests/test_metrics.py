import jax.numpy as jnp
from scipy.spatial.distance import dice

from src.medseg.networks import dice as mydice


def test_dice():

    def dice_eval(a1, a2):
        return dice(a1, a2), mydice(jnp.array(a1), jnp.array(a2))

    dist1, mydist1 = dice_eval([1, 0, 0], [0, 1, 0])
    assert jnp.allclose(dist1, 1.0)
    assert jnp.allclose(mydist1, dist1)

    dist2, mydist2 = dice_eval([1, 0, 0], [1, 1, 0])
    assert jnp.allclose(dist2, 1 / 3)
    assert jnp.allclose(mydist2, dist2)

    dist3, mydist3 = dice_eval([1, 0, 0], [2, 0, 0])
    assert jnp.allclose(dist3, -1 / 3)
    assert jnp.allclose(mydist3, dist3)
