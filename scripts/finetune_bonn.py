
import pickle
import os
import jax
from datetime import datetime

import jax.numpy as jnp
import numpy as np
import optax
from clu import metric_writers
import matplotlib.pyplot as plt
from src.medseg.data_loader import Loader
from flax import linen as nn
from tqdm import tqdm
import SimpleITK as sitk  # noqa: N813
import skimage
from sklearn.metrics import jaccard_score
import pickle
from flax.core.frozen_dict import FrozenDict

from src.medseg.util import resample_image, compute_roi
from src.medseg.train import UNet3D, save_network, normalize
from scripts.sample_bonn import store_img_in_dict
from src.medseg.util import softmax_focal_loss

def read_folder(path: str):
    reader = sitk.ImageSeriesReader()
    # sitk does not understand path objects.
    dicom_names = reader.GetGDCMSeriesFileNames(str(path))
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    return image

def load_bonn(path: str):
    images = {}
    
    for root, _, filenames in tqdm(os.walk(path), desc="Loading data."):
        folder = root.split("/")[-1]
        patient_id = list(filter(lambda s: s.isdigit(), root.split("/")))
        if patient_id:
            if len(patient_id) > 1:
                # gets rid of the 1000.
                patient_id = patient_id[1]
            else:
                patient_id = patient_id[0]
        if 'T2_cor' in folder:
            cor = read_folder(root)
            store_img_in_dict(images, cor, patient_id, 'cor')
        elif 'T2_sag' in folder:
            sag = read_folder(root)
            store_img_in_dict(images, sag, patient_id, 'sag')
        elif 'T2_tra' in folder:
            tra = read_folder(root)
            store_img_in_dict(images, tra, patient_id, 'tra')

        if 'Zones.nii.gz' in filenames:
            annos = sitk.ReadImage(f'{root}/Zones.nii.gz')
            store_img_in_dict(images, annos, patient_id, 'annos')

    return images


def get_roi(images, input_shape):
        # Resample
        t2w_img = resample_image(images['tra'], [0.5, 0.5, 3.0], sitk.sitkLinear, 0)
        sag_img = resample_image(images['sag'], [0.5, 0.5, 3.0], sitk.sitkLinear, 0)
        annos = resample_image(images['annos'], [0.5, 0.5, 3.0], sitk.sitkLinear, 0)
        # cor_img = resample_image(cor_img, [0.5, 0.5, 3.0], sitk.sitkLinear, 0)
        # annos = resample_image(annos, [0.5, 0.5, 3.0], sitk.sitkLinear, 0)

        regions, slices = compute_roi((t2w_img, sag_img))
        annos_array = sitk.GetArrayFromImage(annos).transpose((1, 2, 0))
        anno_roi = annos_array[tuple(slices[0])]

        # test_t2w = sitk.GetArrayFromImage(t2w_img).transpose((1, 2, 0))[tuple(slices[0])]

        # resample
        resample = True
        if resample:
            t2w_roi = skimage.transform.resize(regions[0], input_shape)
            annos_roi = skimage.transform.resize(anno_roi, input_shape)

        return {"image": t2w_roi, "annos": annos_roi}





if __name__ == "__main__":

    # parameters
    epochs = 100
    input_shape = [128, 128, 21]
    val_keys = ['129', '091']
    # log writer
    writer = metric_writers.create_default_writer(f"./temp/log/{datetime.now()}")

    path = '/home/admin_ml/Barbara/localMRF/Project_MRF/Project_MRF/'
    bonn_scans = load_bonn(path)

    with open("./temp/img.pkl", "wb") as f:
        pickle.dump(bonn_scans, f)

    bonn_rois = {}
    # extract rois
    for id, images in bonn_scans.items():
        # use exclusively patients with annotations.
        if 'annos' in images.keys():
            bonn_rois[id] = get_roi(images, input_shape)

    # set up validation
    bonn_rois_val = {}
    for val_key in val_keys:
        bonn_rois_val[val_key] = bonn_rois[val_key]
        del bonn_rois[val_key]
        assert val_key not in bonn_rois
        assert val_key in bonn_rois_val 

    # validation stack
    val_x, val_y = zip(*[(brv['image'], brv['annos']) for _, brv in bonn_rois_val.items()])
    val_x = jnp.stack(val_x)
    val_y = jnp.stack(val_y)

    model = UNet3D()
    with open("./weights/unet_499.pkl", "rb") as f:
        net_state = pickle.load(f)



    opt = optax.adam(0.001)
    opt_state = opt.init(net_state)
    img_stack = jnp.stack([br['image'] for _, br in bonn_rois.items()])
    mean = jnp.mean(img_stack)
    std = jnp.std(img_stack)

    # normalize validation data
    val_x = normalize(val_x, mean=mean, std=std)

    @jax.jit
    def forward_step(
        variables: FrozenDict,
        img_batch: jnp.ndarray,
        label_batch: jnp.ndarray,
    ):
        """Do a forward step."""
        labels = nn.one_hot(label_batch, 5)
        out = model.apply(variables, img_batch)
        # loss = jnp.mean(
        #      optax.softmax_cross_entropy(logits=out, labels=labels)
        # )
        # loss = jnp.mean(sigmoid_focal_loss(
        #    logits=out, labels=labels, alpha=-1, gamma=2))
        loss = jnp.mean(softmax_focal_loss(out, labels, np.ones([out.shape[-1]])))
        return loss
    
    loss_grad = jax.value_and_grad(forward_step)

    iteration = 0
    bar = tqdm(range(epochs))
    for e in bar:
        for key, xy in bonn_rois.items():
            x, y = xy['image'], xy['annos']
            x = normalize(x, mean=mean, std=std)
            # add batch.
            x = jnp.expand_dims(x, 0)
            cost, grad = loss_grad(net_state, x, y)
            updates, opt_state = opt.update(grad, opt_state, net_state)
            net_state = optax.apply_updates(net_state, updates)
            
            bar.set_description(f"i: {iteration}, e: {e}, loss: {cost}")

            writer.write_scalars(iteration, {"training_loss": cost})
            if iteration % 10 == 0:
                # validate
                val_cost = forward_step(net_state, val_x, val_y)
                net_out = model.apply(net_state, val_x)
                net_labels = jnp.argmax(net_out, -1)
                acc = jnp.sum(net_labels == val_y)/np.prod(list(net_labels.shape))
                iou = jaccard_score(val_y.flatten().astype(int), net_labels.flatten(), average='weighted')
                writer.write_scalars(iteration, {"val_acc_e_{e}": acc, "val_cost": val_cost, "val_iou": iou, "e": e})

                overlay_0 = val_x[0]/jnp.max(val_x[0]) + net_labels[0]/jnp.max(net_labels[0])
                writer.write_images(iteration, {f"val_sample_{val_keys[0]}": 
                                                jnp.expand_dims(jnp.expand_dims(overlay_0[:, :, 12], 0), -1)})
                
            iteration += 1
            
    print('done')
    pass
