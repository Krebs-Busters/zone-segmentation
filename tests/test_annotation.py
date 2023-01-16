import medpy.io
import glob
import numpy as np
import matplotlib.pyplot as plt

def test_loading_human_expert_annotations():
    files = glob.glob("data/picai_labels/csPCa_lesion_delineations/human_expert/resampled/*.nii.gz")
    assert len(files) == 1295

    zero_max_counter = 0
    shape_list = []
    channel_list = []
    for file in files:
        niigz = medpy.io.load(file)
        max = (np.max(niigz[0]))
        if max == 0:
            zero_max_counter += 1
        
        shape = niigz[0].shape
        # images are square!
        assert shape[0] == shape[1]
        shape_list.append(shape[0])
        channel_list.append(shape[-1])
    
    if 0:
        plt.hist(shape_list)
        plt.show()
        plt.hist(channel_list)
        plt.show()
        pass