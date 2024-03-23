"""Samples the test data, to asses segmentation quality."""

import pickle
import os
from typing import Dict

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from src.medseg.data_loader import Loader
from flax import linen as nn
from tqdm import tqdm
import SimpleITK as sitk  # noqa: N813
import skimage

from src.medseg.train import UNet3D, normalize
from src.medseg.util import resample_image, compute_roi

def read_folder(path: str):
    reader = sitk.ImageSeriesReader()
    # sitk does not understand path objects.
    dicom_names = reader.GetGDCMSeriesFileNames(str(path))
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    return image


def store_img_in_dict(dict, image, patient_id, img_key):
    if not patient_id in dict:
        dict[patient_id] = {}
    dict[patient_id][img_key] = image

def load_bonn(path: str):
    images = {}
    
    for root, _, _ in tqdm(os.walk(path), desc="Loading data."):
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

    return images


def get_roi(t2w_img, sag_img, input_shape):
        # Resample
        t2w_img = resample_image(t2w_img, [0.5, 0.5, 3.0], sitk.sitkLinear, 0)
        sag_img = resample_image(sag_img, [0.5, 0.5, 3.0], sitk.sitkLinear, 0)
        regions, _ = compute_roi((t2w_img, sag_img))
        # resample
        resample = True
        if resample:
            t2w_roi = skimage.transform.resize(regions[0], input_shape)
        return t2w_roi



if __name__ == "__main__":

    def disp_result(
        data: jnp.ndarray, out: jnp.ndarray, slice: int = 11
    ):
        """Plot the original image, network output and annotation."""
        data_pick = data[:, :, slice]
        net_seg_pick = jnp.argmax(out[0, :, :, slice], axis=-1)
        plt.title("overlay")
        plt.imshow(data_pick/jnp.max(data_pick) + net_seg_pick/jnp.max(net_seg_pick))
        plt.show()

    mean = jnp.array([206.12558])
    std = jnp.array([164.74423])
    input_shape = [128, 128, 21]

    # remove 001 for all scans.
    # path = '/home/admin_ml/Barbara/localMRF/Project_MRF/test/171'
    path = '/run/user/1000/gvfs/smb-share:server=klinik.bn,share=nas,user=bwichtmann/RAD-MRT/Barbara_Fingerprint/Project_MRF_2/'

    bonn_scans = load_bonn(path)
    bonn_rois = {}
    # extract rois
    for id, images in bonn_scans.items():
        bonn_rois[id] = get_roi(images['tra'], images['sag'], input_shape)

   
    # network = "./weights/unet_499.pkl"
    network = "./weights/unet_tuned_99.pkl"

    model = UNet3D()
    with open(network, "rb") as f:
        net_state = pickle.load(f)

    net_segs = {}
    for id, roi in bonn_rois.items():
        norm_roi = normalize(jnp.array(roi), mean=mean, std=std)
        val_out = model.apply(net_state, jnp.expand_dims(norm_roi, 0))
        # disp_result(roi, val_out)
        net_segs[id] = jnp.argmax(val_out[0], axis=-1)


    # export segmentation
    to_export = {}
    for id in net_segs.keys():
        target_shape = bonn_scans[id]['tra'].GetSize()
        target_space = bonn_scans[id]['tra'].GetSpacing()
        target_origin = bonn_scans[id]['tra'].GetOrigin()
        target_rotation = bonn_scans[id]['tra'].GetDirection()
        resized = skimage.transform.resize(net_segs[id], target_shape)
        export_image = sitk.GetImageFromArray(resized)
        export_image = resample_image(export_image, target_space, sitk.sitkLinear, 0)
        export_image.SetOrigin(target_origin)
        export_image.SetDirection(target_rotation)
        to_export[id] = export_image

        sitk.WriteImage(export_image, f'./temp/export/{id}_annotation.nii.gz')

    print('done')
