import SimpleITK as sitk

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from . import zone_segmentation_utils as utils


def box_lines(size, start=[0, 0, 0]):
    stop = start + size
    bc = np.array([start[0],  start[1], start[2]])
    br = np.array([stop[0],   start[1], start[2]])
    bl = np.array([start[0],  stop[1],  start[2]])
    bb = np.array([stop[0],   stop[1],  start[2]])
    tc = np.array([start[0],  start[1], stop[2]])
    tr = np.array([stop[0],   start[1], stop[2]])
    tl = np.array([start[0],  stop[1],  stop[2]])
    tb = np.array([stop[0],   stop[1],  stop[2]])
    lines = [np.linspace(bc, br, 100),
             np.linspace(br, bb, 100),
             np.linspace(bb, bl, 100),
             np.linspace(bl, bc, 100),
             np.linspace(bb, tb, 100),
             np.linspace(bl, tl, 100),
             np.linspace(bc, tc, 100),
             np.linspace(br, tr, 100),
             np.linspace(tc, tr, 100),
             np.linspace(tr, tb, 100),
             np.linspace(tb, tl, 100),
             np.linspace(tl, tc, 100)]
    return lines

def compute_roi(images):
    assert len(images) == 3

    origins = [np.asarray(img.GetOrigin()) for img in images]
    sizes = [np.asarray(img.GetSpacing())*np.asarray(img.GetSize()) for img in images]
    orientations = [np.asarray(img.GetDirection()).reshape(3, 3) for img in images]

    rects = []
    for pos, size in enumerate(sizes):
        lines = box_lines(size)
        rotated = [(orientations[pos]@line.T).T for line in lines]
        shifted = [origins[pos]+line for line in rotated]
        rects.append(shifted)

    # find the intersection.
    rects = np.stack(rects)
    bbs = [(np.amin(rect, axis=(0,1)), np.amax(rect, axis=(0,1))) for rect in rects]

    # compute intersection
    lower_end = np.amax(np.stack([bb[0] for bb in bbs], axis=0), axis=0)
    upper_end = np.amin(np.stack([bb[1] for bb in bbs], axis=0), axis=0)
    roi_bb = np.stack((lower_end, upper_end))
    roi_bb_size = roi_bb[1] - roi_bb[0]

    roi_bb_lines = np.stack(box_lines(roi_bb_size, roi_bb[0]))
    rects = np.concatenate([rects, np.expand_dims(roi_bb_lines, 0)])

    spacings = [image.GetSpacing() for image in images]
    # compute roi coordinates in image space.
    img_coord_rois = [((np.linalg.inv(rot)@(roi_bb[0] - offset).T).T / spacing,
                       (np.linalg.inv(rot)@(roi_bb[1] - offset).T).T / spacing)
                         for rot, offset, spacing in zip(orientations, origins, spacings)]


    arrays = [sitk.GetArrayFromImage(image).transpose((1, 2, 0)) for image in images]
    box_indices = []
    for ib, array in zip(img_coord_rois, arrays):
        img_indices = []
        low, up = np.amin(ib, axis=0), np.amax(ib, axis=0)
        for pos, dim in enumerate(array.shape):
            def in_array(in_int, dim):
                in_int = int(in_int)
                in_int = 0 if in_int < 0 else in_int
                in_int = dim if in_int > dim else in_int
                return in_int
            img_indices.append(slice(in_array(low[pos], dim),
                                     in_array(up[pos], dim)))
        box_indices.append(img_indices)
    
    intersections = [i[tuple(box_inds)] for box_inds, i in zip(box_indices, arrays)]

    if False:
        # plot rects
        names = ['tra', 'cor', 'sag', 'roi']
        color_keys = list(mcolors.TABLEAU_COLORS.keys())
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        for pos, rect in enumerate(rects):
            color = color_keys[pos%len(color_keys)]
            for linepos, line in enumerate(rect):
                if linepos == 0:
                    ax.plot(line[0, 0], line[0, 1], line[0 , 2] , 's', color=color, label=names[pos])
                ax.plot(line[:, 0], line[:, 1], line[:, 2] , '-.', color=color)
        plt.legend()
        plt.show()

        for pos, rect in enumerate(rects):
            color = color_keys[pos%len(color_keys)]
            for linepos, line in enumerate(rect):
                if linepos == 0:
                    plt.plot(line[0, 0], line[0, 1], 's', color=color, label=names[pos])
                plt.plot(line[:, 0], line[:, 1], '-.', color=color)
        plt.legend(loc='upper right')
        plt.title('X,Y-View')
        plt.show()

        for pos, rect in enumerate(rects):
            color = color_keys[pos%len(color_keys)]
            for linepos, line in enumerate(rect):
                if linepos == 0:
                    plt.plot(line[0, 1], line[0, 2], 's', color=color, label=names[pos])
                plt.plot(line[:, 1], line[:, 2], '-.', color=color)
        plt.legend(loc='upper right')
        plt.title('Y-Z-View')
        plt.show()


        # img_coord_tra = img_coord_rois[0]
        # sitk getShape and GetArrayFromImage return transposed results.

        plt.imshow(intersections[0][:, :, 10])
        plt.show()

    return intersections, box_indices




def compute_roi2(images):

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