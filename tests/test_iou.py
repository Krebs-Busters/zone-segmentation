"""Test the IoU compute function."""
import sys

sys.path.insert(0, "./src")

import jax.numpy as jnp
import pytest

from src.data_loader import Loader
from src.meanIoU import compute_iou

input_records = ["ProstateX-0004", "ProstateX-0007", "ProstateX-0311"]


@pytest.mark.offline
@pytest.mark.parametrize("input_record", input_records)
def test_iou(input_record: str) -> None:
    """Test for IoU.

    Args:
        input_record (str): Input record for test
    """
    loader = Loader(input_shape=(128, 128, 21))
    record = loader.get_record(input_record)
    target = jnp.expand_dims(record["annotation"], axis=0)
    iou = compute_iou(target, target)
    assert iou == 1.0
