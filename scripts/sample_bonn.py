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
        # cor_img = resample_image(cor_img, [0.5, 0.5, 3.0], sitk.sitkLinear, 0)
        # annos = resample_image(annos, [0.5, 0.5, 3.0], sitk.sitkLinear, 0)

        regions, _ = compute_roi((t2w_img, sag_img))
        # anneke = compute_roi2((t2w_img, cor_img, sag_img))
        # anneke = sitk.GetArrayFromImage(anneke[0])
        # import matplotlib.pyplot as plt
        # t2w_img_array = sitk.GetArrayFromImage(t2w_img).transpose((1, 2, 0))

        # annos_array = sitk.GetArrayFromImage(annos).transpose((1, 2, 0))
        # anno_roi = annos_array[tuple(slices[0])]

        # test_t2w = sitk.GetArrayFromImage(t2w_img).transpose((1, 2, 0))[tuple(slices[0])]

        # resample
        resample = True
        if resample:
            t2w_roi = skimage.transform.resize(regions[0], input_shape)

        return t2w_roi



if __name__ == "__main__":

    def disp_result(
        data: Dict[str, jnp.ndarray], out: jnp.ndarray, slice: int = 11
    ):
        """Plot the original image, network output and annotation."""
        plt.title("scan")
        plt.imshow(data[:, :, slice])
        plt.show()
        plt.title("network")
        plt.imshow(jnp.argmax(out[0, :, :, slice], axis=-1), vmin=0, vmax=5)
        plt.show()
        # plt.title("human expert")
        # plt.imshow(data["annotation"][sample, :, :, slice], vmin=0, vmax=5)
        # plt.show()
        pass


    mean = jnp.array([206.12558])
    std = jnp.array([164.74423])
    input_shape = [128, 128, 21]

    # remove 001 for all scans.
    path = '/run/user/1000/gvfs/smb-share:server=klinik.bn,share=nas,user=bwichtmann/RAD-MRT/Barbara_Fingerprint/Project_MRF/001'
    bonn_scans = load_bonn(path)
    bonn_rois = {}
    # extract rois
    for id, images in bonn_scans.items():
        bonn_rois[id] = get_roi(images['tra'], images['sag'], input_shape)

   
    model = UNet3D()
    with open("./weights/unet_499.pkl", "rb") as f:
        net_state = pickle.load(f)

    mannheim_net_segs = {}
    for id, roi in bonn_rois.items():
        norm_roi = normalize(jnp.array(roi), mean=mean, std=std)
        val_out = model.apply(net_state, jnp.expand_dims(norm_roi, 0))
        disp_result(roi, val_out)

    pass
