import medpy
import pathlib


class DataLoader(object):

    def __init__(self, data_path: str='./data'):
        self.data_path = pathlib.Path(data_path)
        fold1_files = list(self.data_path.glob('full/picai_public_images_fold0/**/*.mha'))
        annotations = list(self.data_path.glob('full/picai_public_images_fold0/**/*.mha'))
        pass