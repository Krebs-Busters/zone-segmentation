from data_loader import PicaiLoader

import jax
import pickle
from flax.core.frozen_dict import FrozenDict
import jax.numpy as jnp
from train_model import UNet3D, normalize
import matplotlib.pyplot as plt

@jax.jit
def forward_step(variables: FrozenDict, test_images: jnp.ndarray):
    output_mask = model.apply(variables, test_images)
    return output_mask

if __name__ == '__main__':
    weights = pickle.load(open('./weights/picaiunet_adam.pkl', 'rb'))
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
    
    for i in range(val_data_images.shape[-1]):
        plt.figure()

        plt.subplot(131)
        plt.imshow(val_data['images'][0, :, :, i], cmap='gray')
        plt.title(f'Input Image')
        plt.axis('off')

        plt.subplot(132)
        plt.imshow(val_data['annotation'][0, :, :, i], cmap='gray')
        plt.title(f'Annotation')
        plt.axis('off')

        plt.subplot(133)
        plt.imshow(jnp.sum(output_masks, axis=-1)[0, :, :, i], cmap='gray')
        plt.title(f'Network output')
        plt.axis('off')

        plt.suptitle(f'Slice-{i}')
        plt.savefig(f"./images/images_adam/img_sample_{i}.jpg")
        plt.close()