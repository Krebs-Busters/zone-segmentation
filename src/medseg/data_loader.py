import medpy.io
import pathlib
import glob
from multiprocessing import Pool
import numpy as np
import jax.tree_util


class DataLoader(object):

    def __init__(self, data_path: str='./data', fold: int=0, worker_no: int=5):
        self.data_path = pathlib.Path(data_path)
        self.fold = fold
        self.fold_files = list(
            self.data_path.glob(f'full/picai_public_images_fold{self.fold}/**/*.mha'))
        self.annotations = list(
            pathlib.Path("data/picai_labels/csPCa_lesion_delineations/human_expert/resampled/").glob("*.nii.gz"))
        
        self.create_images_dict()
        self.create_annotation_dict()

        self.key_pointer = 0
        self.patient_keys = list(self.image_dict.keys())
        self.worker_no = worker_no


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
        return self.patient_keys[self.key_pointer]

    def get_record(self, patient_key):
        """ Load a patient's image data and annotations. """
        images = self.image_dict[patient_key]
        annos = self.annotation_dict[patient_key]

        def load_dict(path_dict: dict) -> dict:
            image_dict = {}
            for key, object in path_dict.items():
                image_dict[key] = medpy.io.load(object)[0]
            return image_dict

        images = load_dict(images)
        annos = medpy.io.load(annos)[0]

        return {"images": images, "annotation": annos}


    def get_batch(self, batch_size: int):
        patient_keys = [self.get_key() for _ in range(batch_size)]
        batch_dict_list = list(map(self.get_record, patient_keys))

        def tree_stack(trees):
            """Takes a list of trees and stacks every corresponding leaf.
            For example, given two trees ((a, b), c) and ((a', b'), c'), returns
            ((stack(a, a'), stack(b, b')), stack(c, c')).
            Useful for turning a list of objects into something you can feed to a
            vmapped function.

            Cudos to:
            https://gist.github.com/willwhitney/dd89cac6a5b771ccff18b06b33372c75
            """
            leaves_list = []
            treedef_list = []
            for tree in trees:
                leaves, treedef = jax.tree_util.tree_flatten(tree)
                leaves_list.append(leaves)
                treedef_list.append(treedef)

            grouped_leaves = zip(*leaves_list)
            result_leaves = [np.stack(l) for l in grouped_leaves]
            return treedef_list[0].unflatten(result_leaves)

        stacked = tree_stack(batch_dict_list)

        return stacked