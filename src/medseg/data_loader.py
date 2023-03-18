import SimpleITK as sitk
import skimage
import pathlib
import numpy as np
import jax.tree_util

from typing import Tuple, List
from multiprocessing.dummy import Pool

from .util import compute_roi, compute_roi2


class PicaiLoader(object):

    def __init__(self, data_path: str='./data', fold: int=0, workers=None,
                 input_shape: Tuple[int, int, int] = (256, 256, 21),
                 val_keys: List[str] = []):
        """Generate a picai-data loader.

        Args:
            data_path (str, optional): Where to find the Picai data set. Defaults to './data'.
            fold (int, optional): The desired picai fold. Defaults to 0.
            workers (_type_, optional): The number of worker-threads for the data loader.
                Defaults to None, which uses os.cpu_count() .
        """        
        self.data_path = pathlib.Path(data_path)
        self.fold = fold
        self.fold_files = list(
            self.data_path.glob(f'full/picai_public_images_fold{self.fold}/**/*.mha'))
        self.annotations = list(
            pathlib.Path("data/picai_labels/csPCa_lesion_delineations/human_expert/resampled/").glob("*.nii.gz"))
        
        self.create_images_dict()
        self.create_annotation_dict()

        self.key_pointer = 0

        self.patient_keys = [key for key in self.annotation_dict.keys() if key in list(self.image_dict.keys())]
        
        if val_keys:
            self.train_patient_keys = [key for key in self.patient_keys if key not in val_keys]
            self.val_keys = val_keys
        else:
            self.train_patient_keys = self.patient_keys
            self.val_keys = None

        print(f"Found {len(self.train_patient_keys)} matching train keys.")
        self._workers = workers
        self.input_shape = input_shape
        self.reset = False



    def create_annotation_dict(self):
        self.annotation_dict = {}
        for annotation_path in self.annotations:
            patient_id = str(annotation_path).split('/')[-1].split('.')[0]
            self.annotation_dict[patient_id] = annotation_path

    def create_images_dict(self):
        self.image_dict = {}
        for file_path in self.fold_files:
            path_split = str(file_path).split('/')
            patient_id = "_".join(path_split[-1].split('_')[:2])
            file_type = path_split[-1][-7:-4]
            if patient_id in self.image_dict:
                self.image_dict[patient_id][file_type] = file_path 
            else:
                self.image_dict[patient_id] = {file_type: file_path}
        

    def get_key(self):
        if self.key_pointer >= len(self.train_patient_keys):
               self.key_pointer = 0
               self.reset = True
        current_key = self.train_patient_keys[self.key_pointer]
        self.key_pointer += 1 
        return current_key


    def get_record(self, patient_key):
        """ Load a patient's image data and annotations.
        Records should have the following keys:

        t2w: axial t2w,
        sag: saggital t2w,
        cor: coronal t2w,
        hdv: axial dwi
        adc: axial adc 
        """
        images = self.image_dict[patient_key]
        annos = self.annotation_dict[patient_key]

        # roi
        import matplotlib.pyplot as plt
        t2w_img = sitk.ReadImage(images['t2w'])
        sag_img = sitk.ReadImage(images['sag'])
        cor_img = sitk.ReadImage(images['cor'])
        regions, slices = compute_roi((t2w_img, cor_img, sag_img))
        # anneke = compute_roi2((t2w_img, cor_img, sag_img))
        # anneke = sitk.GetArrayFromImage(anneke[0])

        annos = sitk.ReadImage(annos)
        annos = sitk.GetArrayFromImage(annos).transpose((1, 2, 0))
        annos = annos[tuple(slices[0])]

        # test_t2w = sitk.GetArrayFromImage(t2w_img).transpose((1, 2, 0))[tuple(slices[0])]

        # resample
        roi = skimage.transform.resize(regions[0], self.input_shape)
        annos = skimage.transform.resize(annos.astype(np.uint8), self.input_shape, preserve_range=True)
        annos = np.rint(annos)
        # TODO: support all labels.
        annos = np.where(annos > 0.1, np.ones_like(annos), np.zeros_like(annos))
        return {"images": roi, "annotation": annos}

    def stack_samples(self, loaded_samples): 
        leaves_list = []
        treedef_list = []
        for tree in loaded_samples:
            leaves, treedef = jax.tree_util.tree_flatten(tree)
            leaves_list.append(leaves)
            treedef_list.append(treedef)
        grouped_leaves = list(zip(*leaves_list))
        result_leaves = map(np.stack, grouped_leaves)
        stacked = treedef_list[0].unflatten(result_leaves)
        return stacked


    def get_batch(self, batch_size: int):
        patient_keys = [self.get_key() for _ in range(batch_size)]
        # with Pool(self._workers) as p:
        if 1:
            # todo move from map to p.map after debugging.
            batch_dict_list = list(map(self.get_record, patient_keys))
            # batch_dict_list = p.map(self.get_record, patient_keys)
            stacked = self.stack_samples(batch_dict_list)
        return stacked

    def get_epoch(self, batch_size):
        self.reset = False
        while self.reset is False:
            yield self.get_batch(batch_size)

    def get_val(self):
        if self.val_keys:
            val_samples = []
            for val_key in self.val_keys:
                val_samples.append(self.get_record(val_key))
            stacked = self.stack_samples(val_samples)
            return stacked
        else:
            print("Warning: no validation keys found.")
            return None