from data_loader import PicaiLoader

import jax
import pickle
from flax.core.frozen_dict import FrozenDict
import jax.numpy as jnp
from train_model import UNet3D, normalize

@jax.jit
def forward_step(variables: FrozenDict, test_images: jnp.ndarray):
    output_mask = model.apply(variables, test_images)
    return output_mask

if __name__ == '__main__':
    weights = pickle.load(open('./weights/picaiunet_sgd_50.pkl', 'rb'))
    input_shape = [256, 256, 21]
    val_keys = ['10085_1000085', '10730_1000746', '10525_1000535']
    data_mean = jnp.array([206.12558])
    data_std = jnp.array([164.74423])

    dataset = PicaiLoader(input_shape= input_shape, val_keys= val_keys)
    batch_size =  1

    val_data = dataset.get_val(batch_size)[0]
    val_data_images = normalize(val_data['images'], data_mean, data_std)

    model = UNet3D()
    output_masks = model.apply(weights, val_data_images)
    pass