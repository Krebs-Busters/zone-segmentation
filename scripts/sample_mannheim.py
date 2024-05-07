"""Training script for Barbara Wichtmanns in-house 'mannheim'-data set."""

import pickle
import os

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score
import SimpleITK as sitk  # noqa: N813
import skimage
from tqdm import tqdm


from scripts.train_prostate_X import UNet3D, normalize
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
        regions, indcies = compute_roi((t2w_img, cor_img, sag_img))

        # resample
        resample = True
        if resample:
            t2w_roi = skimage.transform.resize(regions[0], input_shape)

        return t2w_roi, indcies[0]


if __name__ == "__main__":

    def disp_result(
        data: jnp.ndarray, labels: jnp.ndarray, id, scan, slice: int = 20
    ):
        """Plot the original image, network output and annotation."""
        colors = [[0., 0., 0.],
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
        # plt.savefig(f"./export/net_seg_{id}_{scan}.png")
        # plt.show()
        if id == '42498984':
            pass
        plt.clf()

    input_shape = [168, 168, 32]

    part = 3
    path = f'/home/wolter/uni/cancer/Skyra_Mannheim/Part{part}/Part{part}/'
    # path = f'/home/wolter/uni/cancer/Skyra_Mannheim/Part{part}/Part{part}/42498984/'
    mannheim_scans = load_mannhein(path)
    mannheim_rois = {}
    # extract rois
    for id, images in mannheim_scans.items():
        tras = {}
        for tra_scan in filter(lambda s: 'tra' in s, images.keys()):
            tras[tra_scan] = get_roi(images[tra_scan], images['sag'], images['cor'], input_shape)
        mannheim_rois[id] = tras

    mean = jnp.mean(jnp.array([tra[0] for _, tra in tras.items()]))
    std = jnp.std(jnp.array([tra[0] for _, tra in tras.items()]))

    model = UNet3D()
    with open("./weights/archive/unet_499.pkl", "rb") as f:
        net_state = pickle.load(f)

    mannheim_net_segs = {}
    jaccard_score_dict = {}
    net_segs = {}
    for id, roi in tqdm(mannheim_rois.items()):
        tras = {}
        for tra_id, tra_roi in mannheim_rois[id].items():
            norm_roi = normalize(jnp.array(tra_roi[0]), mean=mean, std=std)
            val_out = model.apply(net_state, jnp.expand_dims(norm_roi, 0))
            val_out = jnp.argmax(val_out, axis=-1)
            tras[tra_id] = val_out
            net_segs[id] = tras
            disp_result(tra_roi[0], val_out, id, tra_id)
        mannheim_net_segs[id] = tras
        try:
            jaccard_score_dict[id] = jaccard_score(*[labels.flatten() for labels in tras.values()],
                                                   average='weighted')
        except Exception as err:
            print(f"Excpetion at {id}, {tra_id}", err)
    print(f'jaccard scores: {jaccard_score_dict}')

    mean_iou = np.mean([value for _, value in jaccard_score_dict.items()])

    print(f'mean IoU: {mean_iou:2.2f}')
    
    # segmentation export.
    to_export = {}
    for id in net_segs.keys():
        for scan_key in mannheim_rois[id].keys():
            scan_value = mannheim_scans[id][scan_key]
            pre_sample_size = resample_image(scan_value, [0.5, 0.5, 3.0], sitk.sitkLinear, 0).GetSize()
            indices = mannheim_rois[id][scan_key][1]
            pad_segs = np.zeros(pre_sample_size)
            segmentation = net_segs[id][scan_key][0]
            pad_segs[indices[0], indices[1], indices[2]] = skimage.transform.resize(segmentation,
                                                                                    (sl.stop - sl.start for sl in indices))
            target_shape = mannheim_scans[id][scan_key].GetSize()
            target_space = mannheim_scans[id][scan_key].GetSpacing()        
            target_origin = mannheim_scans[id][scan_key].GetOrigin()
            target_rotation = np.array(mannheim_scans[id][scan_key].GetDirection()).reshape(3, 3)
            target_rotation_fix = np.concatenate([target_rotation[1, :], target_rotation[0, :], target_rotation[-1, :]])
            rotation_list = list(target_rotation_fix.flatten())
            export_image = sitk.GetImageFromArray(pad_segs.transpose([2, 1, 0]))
            export_image.SetSpacing([0.5, 0.5, 3.0])
            export_image = resample_image(export_image, target_space, sitk.sitkLinear, 0)
            export_image.SetOrigin(target_origin)
            export_image.SetDirection(rotation_list)
            to_export[id] = export_image

            # export pfad wo die netzwerk segmentierungen gespeichert werden.
            sitk.WriteImage(export_image, f'export/mannheim/{id}_{scan_key}_annotation.nii.gz')