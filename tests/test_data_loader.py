import medpy.io
import pytest
import numpy as np
import jax.numpy as jnp
import jax.nn

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


def test_annotation():
    import matplotlib.pyplot as plt
    loader = PicaiLoader()
    record = loader.get_record('10257_1000261')
    interesting_annotation = record['annotation']
    interesting_annotation_hot = jax.nn.one_hot(interesting_annotation, 2)
    assert np.allclose(interesting_annotation_hot[..., 1], interesting_annotation)
    assert np.allclose(interesting_annotation_hot[..., 0], 1-interesting_annotation)
    pass