import medpy
import medpy.io.header as header
import SimpleITK as sitk

import numpy as np
import zone_segmentation_utils as utils


# def compute_roi(images, heads):
#     """
#     
#     Extension of https://loli.github.io/medpy/_modules/medpy/filter/utilities.html#intersection .
#     Added multi-image and transformation matrix support.
# 
#     Args:
#         images (_type_): _description_
#         heads (_type_): _description_
#     """
# 
#     # oss = [np.asarray(head.get_offset()) for head in heads]
#     oss = [np.asarray(head.get_sitkimage().GetOrigin()) for head in heads]
#     # pss = [np.asarray(head.get_voxel_spacing()) for head in heads]
#     pss = [np.asarray(head.get_sitkimage().GetSpacing()) for head in heads]
#     # tms = [np.asarray(head.get_direction()) for head in heads]
#     tms = [np.asarray(head.get_sitkimage().GetDirection()).reshape(3,3) for head in heads]
#     inv_tm = [np.linalg.pinv(tm) for tm in tms]
#     bbs = []
#     for pos, image in enumerate(images):
#         low = oss[pos]
#         up = np.asarray(image.shape) * pss[pos] + oss[pos]
#         bbs.append((low, up))
#     bbs = np.stack(bbs)
#     
#     # compute image bounding boxes in real-world coordinates
#     # bb0, os0, ps0 = _get_bb(images[0], heads[0])
#     # bb1, os1, ps1 = _get_bb(images[1], heads[1])
#     # bb2, os2, ps2 = _get_bb(images[2], heads[2])
#     # bbs, oss, pss =  _get_bb(images, heads)
# 
#     # compute intersection
#     lower_end = np.amax(np.stack([bb[0] for bb in bbs], axis=0), axis=0)
#     upper_end = np.amin(np.stack([bb[1] for bb in bbs], axis=0), axis=0)
#     ib_ind = np.stack((lower_end, upper_end))
# 
#     # transfer intersection to respective image coordinates image
#     def _coord_transfer(ib, os, ps):
#         return [((ib[0] - os) / np.asarray(ps)).astype(np.int),
#                 ((ib[1] - os) / np.asarray(ps)).astype(np.int)]
#     
#     ibs = np.stack([_coord_transfer(ib_ind, os, ps) for (os, ps) in zip(oss, pss)])
# 
#     # ensure that sub-volumes are of same size (might be affected by rounding errors); only reduction allowed
#     # sizes = np.stack(list((ib_pair[1] - ib_pair[0]) for ib_pair in ibs))
#     sizes = ibs[:, 1, :] - ibs[:, 0, :]
#     min_sizes = np.amin(sizes, 0)
#     diffs = sizes - min_sizes
#     
#     # subtract differences from the top.
#     ibs[:, 1, :] -= diffs
# 
#     # compute new image offsets (in real-world coordinates); averaged to account for rounding errors due to world-to-voxel mapping
#     nos = ibs[:, 0, :] * pss + oss
#     nos = np.mean(nos, axis=0)
# 
#     # intersections = [i[ibs[pos,0]:ibs[pos,1]] for pos,i in enumerate(images)]
#     box_indices = []
#     for ib in ibs:
#         box_indices.append((slice(low, up) for low, up in zip(*ib)))
#     intersections = [i[box_inds] for box_inds, i in zip(box_indices, images)]
#     return intersections, nos


def compute_roi(images):

    img_tra = images[0]
    img_cor = images[1]
    img_sag = images[2]

    # normalize intensities
    print('... normalize intensities ...')
    img_tra, img_cor, img_sag = utils.normalizeIntensitiesPercentile(img_tra, img_cor, img_sag)

    # get intersecting region (bounding box)
    print('... get intersecting region (ROI) ...')

    # upsample transversal image to isotropic voxel size (isotropic transversal image coordinate system is used as reference coordinate system)
    tra_HR = utils.resampleImage(img_tra, [0.5, 0.5, 0.5], sitk.sitkLinear,0)
    # tra_HR = utils.sizeCorrectionImage(tra_HR, factor=6, imgSize=168)
    tra_HR = utils.crop_and_pad_sitk(tra_HR, [168, 168, 168])

    # resample coronal and sagittal to tra_HR space
    # resample coronal to tra_HR and obtain mask (voxels that are defined in coronal image )
    cor_toTraHR = utils.resampleToReference(img_cor, tra_HR, sitk.sitkLinear,-1)
    cor_mask = utils.binaryThresholdImage(cor_toTraHR, 0)

    tra_HR_Float = utils.castImage(tra_HR, sitk.sitkFloat32)
    cor_mask_Float = utils.castImage(cor_mask, sitk.sitkFloat32)
    # mask transversal volume (set voxels, that are defined only in transversal image but not in coronal image, to 0)
    coronal_masked_traHR = sitk.Multiply(tra_HR_Float, cor_mask_Float)

    # resample sagittal to tra_HR and obtain mask (voxels that are defined in sagittal image )
    sag_toTraHR = utils.resampleToReference(img_sag, tra_HR, sitk.sitkLinear,-1)
    sag_mask = utils.binaryThresholdImage(sag_toTraHR, 0)
    # mask sagittal volume
    sag_mask_Float = utils.castImage(sag_mask, sitk.sitkFloat32)

    # masked image contains voxels, that are defined in tra, cor and sag images
    maskedImg = sitk.Multiply(sag_mask_Float, coronal_masked_traHR)
    boundingBox = utils.getBoundingBox(maskedImg)

    # correct the size and start position of the bounding box according to new size
    start, size = utils.sizeCorrectionBoundingBox(boundingBox, newSize=168, factor=6)
    start[2] = 0
    start = list(s if s > 0 else 0 for s in start)
    size[2] = tra_HR.GetSize()[2]

    # resample cor and sag to isotropic transversal image space
    cor_traHR = utils.resampleToReference(img_cor, tra_HR, sitk.sitkLinear, -1)
    sag_traHR = utils.resampleToReference(img_sag, tra_HR, sitk.sitkLinear,-1)

    ## extract bounding box for all planes
    region_tra = sitk.RegionOfInterest(tra_HR, [size[0], size[1], size[2]],
                                       [start[0], start[1], start[2]])
    maxVal = utils.getMaximumValue(region_tra)
    region_tra = utils.thresholdImage(region_tra, 0, maxVal, 0)

    region_cor = sitk.RegionOfInterest(cor_traHR, [size[0], size[1], size[2]],
                                       [start[0], start[1], start[2]])
    maxVal = utils.getMaximumValue(region_cor)
    region_cor = utils.thresholdImage(region_cor, 0, maxVal, 0)

    region_sag = sitk.RegionOfInterest(sag_traHR, [size[0], size[1], size[2]],
                                       [start[0], start[1], start[2]])
    maxVal = utils.getMaximumValue(region_sag)
    region_sag = utils.thresholdImage(region_sag, 0, maxVal, 0)

    return region_tra, start, size