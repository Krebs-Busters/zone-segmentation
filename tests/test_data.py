"""Test the python function from src."""
import sys

import pytest
import SimpleITK as sitk  # noqa: N813
from jax.config import config

config.update("jax_platform_name", "cpu")
sys.path.insert(0, "./src/")

from src.data_loader import Loader
from src.util import compute_roi, resample_image


@pytest.mark.offline
def test_data() -> None:
    """See it the function really returns true."""
    loader = Loader()
    test_anno_raw = []
    for key in loader.patient_keys:
        test_anno_raw.append(key in loader.patient_series_dict.keys())
    assert all(test_anno_raw)
    test_raw_complete = []
    interesting_scans = ["t2_tse_tra", "t2_tse_sag", "t2_tse_cor"]
    for key in loader.patient_keys:
        for scan in interesting_scans:
            entry = loader.patient_series_dict[key]
            test_raw_complete.append(scan in entry.keys())
    assert all(test_raw_complete)


@pytest.mark.offline
def test_batch_assembly() -> None:
    """Ensure scans and annotations have the same size."""
    loader = Loader()
    record = loader.get_record("ProstateX-0311")
    assert record["images"].shape == record["annotation"].shape

    batch = loader.get_batch(8)
    assert batch["images"].shape == batch["annotation"].shape


@pytest.mark.offline
def test_roi() -> None:
    """Ensure the region of interest computation works as intendet."""
    loader = Loader()
    t2w_img, sag_img, cor_img = loader.get_images("ProstateX-0311")
    t2w_img = resample_image(t2w_img, [0.5, 0.5, 3.0], sitk.sitkLinear, 0)
    sag_img = resample_image(sag_img, [0.5, 0.5, 3.0], sitk.sitkLinear, 0)
    cor_img = resample_image(cor_img, [0.5, 0.5, 3.0], sitk.sitkLinear, 0)

    _, slices = compute_roi((t2w_img, cor_img, sag_img))

    assert slices[0][0] == slice(102, 277, None)
