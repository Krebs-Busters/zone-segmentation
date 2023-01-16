import pytest
import numpy as np

from src.medseg.data_loader import PicaiLoader


@pytest.mark.parametrize("batch_size", [8, 16])
def test_loader(batch_size: int):
    loader = PicaiLoader()
    # rec = loader.get_record()
    batch = loader.get_batch(batch_size)
    assert list(batch['images']['adc'].shape) == [batch_size] + list(loader.input_shape)
    assert list(batch['images']['sag'].shape) == [batch_size] + list(loader.input_shape)
    assert list(batch['images']['hbv'].shape) == [batch_size] + list(loader.input_shape)
    assert list(batch['images']['cor'].shape) == [batch_size] + list(loader.input_shape)
    assert list(batch['images']['t2w'].shape) == [batch_size] + list(loader.input_shape)
    assert list(batch['annotation'].shape) == [batch_size] + list(loader.input_shape)

    # for i in range(10):
    #     print(i)
    #     batch = loader.get_batch(10)
    #     if np.max(batch['annotation']) > 0.1:
    #         print('hit!')


if __name__ == '__main__':
    # test_loader(128)
    test_loader(12)