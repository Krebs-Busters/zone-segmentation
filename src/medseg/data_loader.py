import medpy.io
# import medpy.filter.image
import skimage
import pathlib
import numpy as np
import jax.tree_util

from typing import Tuple
from multiprocessing.dummy import Pool

class PicaiLoader(object):

    def __init__(self, data_path: str='./data', fold: int=0, workers=None,
                 input_shape: Tuple[int, int, int] = (256, 256, 21)):
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
        print(f"Found {len(self.patient_keys)} matching keys.")
        self._workers = workers
        self.input_shape = input_shape

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
        if self.key_pointer > len(self.patient_keys):
               self.key_pointer == 0
        current_key = self.patient_keys[self.key_pointer]
        self.key_pointer += 1 
        return current_key

    def get_record(self, patient_key):
        """ Load a patient's image data and annotations. """
        images = self.image_dict[patient_key]
        annos = self.annotation_dict[patient_key]

        def load_dict(path_dict: dict) -> dict:
            image_dict = {}
            for key, object in path_dict.items():
                image, header = medpy.io.load(object)
                # image, _ = medpy.filter.image.resample(image, header, (0.5, 0.5, 3.))
                # TODO: ROI extraction!!
                image = skimage.transform.resize(image, self.input_shape)
                image_dict[key] = image
            return image_dict

        images = load_dict(images)
        annos, annohead = medpy.io.load(annos)
        # annos, _ = medpy.filter.image.resample(annos, annohead, (0.5, 0.5, 3.))
        annos = skimage.transform.resize(annos, self.input_shape)
        return {"images": images, "annotation": annos}


    def get_batch(self, batch_size: int):
        patient_keys = [self.get_key() for _ in range(batch_size)]
        with Pool(self._workers) as p:
            # todo move from map to p.map after debugging.
            batch_dict_list = list(map(self.get_record, patient_keys))
            leaves_list = []
            treedef_list = []
            for tree in batch_dict_list:
                leaves, treedef = jax.tree_util.tree_flatten(tree)
                leaves_list.append(leaves)
                treedef_list.append(treedef)
            grouped_leaves = list(zip(*leaves_list))
            result_leaves = p.map(np.stack, grouped_leaves)
            stacked = treedef_list[0].unflatten(result_leaves)
        return stacked