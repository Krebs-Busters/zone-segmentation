"""Ensure the data loading works as expected."""

import jax.nn
import medpy.io
import numpy as np
import pytest

from src.medseg.data_loader import Loader


@pytest.mark.parametrize("batch_size", [8, 16])
def test_loader(batch_size: int):
    """Try to load a batch."""
    loader = Loader()
    # rec = loader.get_record()
    batch = loader.get_batch(batch_size)
    assert list(batch["images"].shape) == [batch_size] + list(loader.input_shape)
    assert list(batch["annotation"].shape) == [batch_size] + list(loader.input_shape)


def test_annotation():
    """This test takes a look at the annotations."""
    loader = Loader()
    # record = loader.get_record('10257_1000261')
    record = loader.get_record("ProstateX-0044")
    interesting_annotation = record["annotation"]
    assert np.allclose(np.max(interesting_annotation), 4.0)


def test_stats():
    loader = Loader()
    data_stack = np.concatenate([b["images"] for b in loader.get_epoch(2)], 0)
    mean = np.mean(data_stack)
    std = np.std(data_stack)
    assert np.allclose(mean, 248.29199)
    assert np.allclose(std, 159.64618)


def test_test_data():
    out_shape = (20, 256, 256, 32)
    loader = Loader()
    test_set = loader.get_test_set()
    assert test_set["annotation"].shape == out_shape
    assert test_set["images"].shape == out_shape
