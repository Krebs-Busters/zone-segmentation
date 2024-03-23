"""Samples the test data, to asses segmentation quality."""

import pickle
import os

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score
import SimpleITK as sitk  # noqa: N813
import skimage
from tqdm import tqdm


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

def load_mannhein(path: str):
    images = {}
    
    for root, _, _ in os.walk(path):
        folder = root.split("/")[-1]
        patient_id = list(filter(lambda s: s.isdigit(), root.split("/")))
        if patient_id:
            patient_id = patient_id[0]
        if 't2_tse_cor' in folder:
            cor = read_folder(root)
            store_img_in_dict(images, cor, patient_id, 'cor')
        elif 't2_tse_sag' in folder:
            sag = read_folder(root)
            store_img_in_dict(images, sag, patient_id, 'sag')
        elif 't2_tse_tra' in folder:
            tra = read_folder(root)
            store_img_in_dict(images, tra, patient_id, folder.split("_")[-1])

    return images


def get_roi(t2w_img, sag_img, cor_img, input_shape):
        # Resample
        t2w_img = resample_image(t2w_img, [0.5, 0.5, 3.0], sitk.sitkLinear, 0)
        sag_img = resample_image(sag_img, [0.5, 0.5, 3.0], sitk.sitkLinear, 0)
        cor_img = resample_image(cor_img, [0.5, 0.5, 3.0], sitk.sitkLinear, 0)
        regions, _ = compute_roi((t2w_img, cor_img, sag_img))

        # resample
        resample = True
        if resample:
            t2w_roi = skimage.transform.resize(regions[0], input_shape)

        return t2w_roi


if __name__ == "__main__":

    def disp_result(
        data: jnp.ndarray, labels: jnp.ndarray, id, scan, slice: int = 11
    ):
        """Plot the original image, network output and annotation."""
        colors = [[0.2422, 0.1504, 0.6603],
                  [0.2647, 0.403, 0.9935],
                  [0.1085, 0.6669, 0.8734],
                  [0.2809, 0.7964, 0.5266],
                  [0.9769, 0.9839, 0.0805]]

        plt.title(f"net_seg_{id}_{scan}")
        data = (data[:, :, slice]/jnp.max(data))*255
        data = jnp.stack([data, data, data], -1)
        labels = labels[0, :, :, slice]
        color_labels = [list(map(lambda idx: colors[idx], row)) for row in labels]
        color_labels = np.stack(color_labels)*255

        mix = ((data + color_labels)/2.).astype(np.uint8)

        plt.imshow(mix)
        plt.savefig(f"./export/net_seg_{id}_{scan}.png")
        plt.clf()

    input_shape = [128, 128, 21]

    part = 2
    path = f'/home/wolter/uni/cancer/Skyra_Mannheim/Part{part}/Part{part}'
    mannheim_scans = load_mannhein(path)
    mannheim_rois = {}
    # extract rois
    for id, images in mannheim_scans.items():
        tras = {}
        for tra_scan in filter(lambda s: 'tra' in s, images.keys()):
            tras[tra_scan] = get_roi(images[tra_scan], images['sag'], images['cor'], input_shape)
        mannheim_rois[id] = tras

    
    # mean = jnp.array([206.12558])
    # std = jnp.array([164.74423])
    mean = jnp.mean(jnp.array([tra for _, tra in tras.items()]))
    std = jnp.std(jnp.array([tra for _, tra in tras.items()]))

    model = UNet3D()
    with open("./weights/unet_499.pkl", "rb") as f:
        net_state = pickle.load(f)

    mannheim_net_segs = {}
    jaccard_score_dict = {}
    for id, roi in tqdm(mannheim_rois.items()):
        tras = {}
        for tra_id, tra_roi in mannheim_rois[id].items():
            norm_roi = normalize(jnp.array(tra_roi), mean=mean, std=std)
            val_out = model.apply(net_state, jnp.expand_dims(norm_roi, 0))
            val_out = jnp.argmax(val_out, axis=-1)
            tras[tra_id] = val_out
            disp_result(tra_roi, val_out, id, tra_id)
        mannheim_net_segs[id] = tras
        jaccard_score_dict[id] = jaccard_score(*[labels.flatten() for labels in tras.values()],
                                               average='weighted')
        pass
    print(jaccard_score_dict)
