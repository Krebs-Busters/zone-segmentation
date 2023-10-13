"""ProstateX loading and preprocessing code."""

import pathlib
import pickle
from functools import partial
from typing import Dict, List, Tuple, Union

import jax.tree_util
import numpy as np
import SimpleITK as sitk  # noqa: N813
import skimage

from util import compute_roi  # , compute_roi2
from util import resample_image


class Loader(object):
    """Load the prostateX data-set. See https://prostatex.grand-challenge.org/ ."""

    empty_list: List[str] = []

    def __init__(
        self,
        data_path: str = "./data",
        input_shape: Tuple[int, int, int] = (256, 256, 32),
        val_keys: List[str] = empty_list,
    ):
        """Generate a picai-data loader.

        Args:
            data_path (str): Where to find the Picai data set. Defaults to './data'.
            input_shape (Tuple[int, int, int]): Size of the 3d input-scan tensor.
            val_keys (List[str]): The training set keys used for validation.
        """
        self.data_path = pathlib.Path(data_path)
        with open(f"{data_path}/scan_index.pkl", "rb") as file_index:
            self.file_index = pickle.load(file_index)

        self.interesting_protocols = ["t2_tse_tra", "t2_tse_sag", "t2_tse_cor"]
        # assemble file-tree
        self.patient_series_dict: Dict[str, Dict[str, str]] = {}
        for entry in self.file_index:
            id = entry["PatientID"]
            seriesuid = entry["SeriesInstanceUID"]
            try:
                protocol = entry["ProtocolName"]
                if id not in self.patient_series_dict.keys():
                    self.patient_series_dict[id] = {}

                self.patient_series_dict[id][protocol] = seriesuid
            except KeyError:
                pass

        self.annotations_train = list(
            pathlib.Path("./data/gtexport/Train/").glob("*/*.nrrd")
        )

        self.annotations_test = list(
            pathlib.Path("./data/gtexport/Test/").glob("*/*.nrrd")
        )

        self._create_annotation_dict()
        self.scan_folders = list(pathlib.Path("./data/tciaDownload/").glob("1.3.6.*"))
        self._create_images_dict()
        self.key_pointer = 0
        self.patient_keys = list(self.annotation_dict.keys())

        if val_keys:
            self.train_patient_keys = [
                key for key in self.patient_keys if key not in val_keys
            ]
            self.val_keys = val_keys
        else:
            self.train_patient_keys = self.patient_keys
            self.val_keys = None

        print(f"Found {len(self.train_patient_keys)} matching train keys.")
        self.input_shape = input_shape
        self.reset = False
        self.test_patient_keys = list(self.annotation_test_dict.keys())

    def _create_annotation_dict(self):
        """Gather the keys of train, test and validation patients."""
        self.annotation_dict = {}
        for annotation_path in self.annotations_train:
            patient_id = str(annotation_path).split("/")[-2]
            self.annotation_dict[patient_id] = annotation_path
        self.annotation_test_dict = {}
        for annotation_path in self.annotations_test:
            patient_id = str(annotation_path).split("/")[-2]
            self.annotation_test_dict[patient_id] = annotation_path

    def _create_images_dict(self):
        self.image_dict = {}
        for scan_folder in self.scan_folders:
            reader = sitk.ImageSeriesReader()
            # sitk does not understand path objects.
            dicom_names = reader.GetGDCMSeriesFileNames(str(scan_folder))
            reader.SetFileNames(dicom_names)
            image = reader.Execute()
            sid = str(scan_folder).split("/")[-1]
            self.image_dict[sid] = image

    def get_key(self):
        """Return the next training key."""
        if self.key_pointer >= len(self.train_patient_keys):
            self.key_pointer = 0
            self.reset = True
        current_key = self.train_patient_keys[self.key_pointer]
        self.key_pointer += 1
        return current_key

    def get_images(self, patient_key: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the t2w, sag and cor scan tuple for a corresponding key.

        Args:
            patient_key (str): The prostateX scan key.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple with with the t2w, sag, cor scans,
                in that order.
        """
        t2w = self.image_dict[self.patient_series_dict[patient_key]["t2_tse_tra"]]
        sag = self.image_dict[self.patient_series_dict[patient_key]["t2_tse_sag"]]
        cor = self.image_dict[self.patient_series_dict[patient_key]["t2_tse_cor"]]
        return t2w, sag, cor

    def get_record(self, patient_key: str, test: bool = False) -> Dict[str, np.ndarray]:
        """Load a patient's image data and annotations.

        Records should have the following keys:
        t2w: axial t2w,
        sag: saggital t2w,
        cor: coronal t2w.

        Args:
            patient_key (str): The ProstateX scan key.
            test (bool): Use test data. Defaults to False.

        Returns:
            Dict[str, np.ndarray]: A dictionary with an
                "images" and "annotation" key.
        """
        t2w_img, sag_img, cor_img = self.get_images(patient_key)
        if test:
            annos = self.annotation_test_dict[patient_key]
        else:
            annos = self.annotation_dict[patient_key]

        # roi
        annos = sitk.ReadImage(annos)

        # Resample
        t2w_img = resample_image(t2w_img, [0.5, 0.5, 3.0], sitk.sitkLinear, 0)
        sag_img = resample_image(sag_img, [0.5, 0.5, 3.0], sitk.sitkLinear, 0)
        cor_img = resample_image(cor_img, [0.5, 0.5, 3.0], sitk.sitkLinear, 0)
        annos = resample_image(annos, [0.5, 0.5, 3.0], sitk.sitkLinear, 0)

        regions, slices = compute_roi((t2w_img, cor_img, sag_img))
        # anneke = compute_roi2((t2w_img, cor_img, sag_img))
        # anneke = sitk.GetArrayFromImage(anneke[0])
        # import matplotlib.pyplot as plt
        # t2w_img_array = sitk.GetArrayFromImage(t2w_img).transpose((1, 2, 0))

        annos_array = sitk.GetArrayFromImage(annos).transpose((1, 2, 0))
        anno_roi = annos_array[tuple(slices[0])]

        # test_t2w = sitk.GetArrayFromImage(t2w_img).transpose((1, 2, 0))[tuple(slices[0])]

        # resample
        resample = True
        if resample:
            t2w_roi = skimage.transform.resize(regions[0], self.input_shape)
            anno_roi = skimage.transform.resize(
                anno_roi.astype(np.uint8), self.input_shape, preserve_range=True
            )
        else:
            t2w_roi = regions[0]
        anno_roi = np.rint(anno_roi)
        return {"images": t2w_roi, "annotation": anno_roi}

    def _stack_samples(
        self, loaded_samples: List[Dict[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """Stacks individual dictionaryies into a single dictionary with batched arrays.

        Args:
            loaded_samples (List[Dict[str, np.ndarray]]): A list with loaded samples.

        Returns:
            Dict[str, np.ndarray]: A dict with batched scans and annotations under the
                'images', and 'annotation' keys respectively.
        """
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

    def get_test_set(self):
        """Return the test set scans and annotations as a batched arrays in a dict."""
        dict_list = list(
            map(partial(self.get_record, test=True), self.test_patient_keys)
        )
        # dict_list = p.map(self.get_record, patient_keys)
        stacked = self._stack_samples(dict_list)
        return stacked

    def get_batch(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Return a batch of training scans and annotations."""
        patient_keys = [self.get_key() for _ in range(batch_size)]
        # with Pool(self._workers) as p:
        # todo move from map to p.map after debugging.
        batch_dict_list = list(map(self.get_record, patient_keys))
        # batch_dict_list = p.map(self.get_record, patient_keys)
        if len(batch_dict_list) > 1:
            stacked = self._stack_samples(batch_dict_list)
        else:
            stacked = {
                key: np.expand_dims(el, 0) for key, el in batch_dict_list[0].items()
            }
        return stacked

    def get_epoch(self, batch_size: int):
        """Return an interator over all training batches.

        Args:
            batch_size (int): The number of samples per batch element.

        Yields:
            Dict[str, np.ndarray]: Dictionaries with
                'images', and 'annotation' keys
        """
        self.reset = False
        while self.reset is False:
            yield self.get_batch(batch_size)

    def get_val(
        self, stack: bool = True, test: bool = False
    ) -> Union[List[Dict[str, np.ndarray]], Dict[str, np.ndarray]]:
        """Get the validation data scans and annotations."""
        if self.val_keys:
            val_samples = []
            for val_key in self.val_keys:
                val_samples.append(self.get_record(val_key, test))
            if stack:
                return self._stack_samples(val_samples)
            else:
                stacked = [
                    {key: np.expand_dims(el, 0) for key, el in val_sample.items()}
                    for val_sample in val_samples
                ]
                return stacked
        else:
            print("Warning: no validation keys found.")
            return None
