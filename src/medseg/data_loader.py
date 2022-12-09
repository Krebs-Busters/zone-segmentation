import medpy.io
import pathlib
import glob
from multiprocessing import Pool


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
                    image_dict[key] = medpy.io.load(object)
            return image_dict

        images = load_dict(images)
        annos = medpy.io.load(annos)

        return {"images": images, "annotation": annos}


    def get_batch(self, batch_size: int):
        images = []
        labels = []
        patient_keys = [self.get_key() for _ in range(batch_size)]

        with Pool(self.worker_no) as p:
            batch_data_dicts = p.map(self.get_record, patient_keys)

        for batch_dict in batch_data_dicts:
            pass

        return None
