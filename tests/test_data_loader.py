import medpy.io
import pytest
import numpy as np

from src.medseg.data_loader import PicaiLoader


@pytest.mark.parametrize("batch_size", [8, 16])
def test_loader(batch_size: int):
    loader = PicaiLoader()
    # rec = loader.get_record()
    batch = loader.get_batch(batch_size)
    assert list(batch['images'].shape) == [batch_size] + list(loader.input_shape)
    assert list(batch['annotation'].shape) == [batch_size] + list(loader.input_shape)



def test_sick_count():
    loader = PicaiLoader()

    # look for people with cancer.
    interesting_list = []
    for key in loader.patient_keys:
        path = loader.annotation_dict[key]
        niigz, _ = medpy.io.load(path)
        max = (np.max(niigz))

        if max >= 0.1:
            interesting_list.append((key, niigz))
    print(len(interesting_list))
    assert len(interesting_list) == 47



