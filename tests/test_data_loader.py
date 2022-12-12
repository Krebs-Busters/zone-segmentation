import pytest
import numpy as np

from src.medseg.data_loader import DataLoader


@pytest.mark.parametrize("batch_size", [5, 15])
def test_loader(batch_size: int):
    loader = DataLoader(worker_no=10)
    # rec = loader.get_record()
    batch = loader.get_batch(batch_size)
    assert batch['images']['adc'].shape == (batch_size, 120, 128, 19)
    assert batch['images']['sag'].shape == (batch_size, 320, 320, 19)
    assert batch['images']['hbv'].shape == (batch_size, 120, 128, 19)
    assert batch['images']['cor'].shape == (batch_size, 320, 320, 17)
    assert batch['images']['t2w'].shape == (batch_size, 640, 640, 19)

    assert batch['annotation'].shape == (batch_size, 640, 640, 19)


    # for i in range(10):
    #     print(i)
    #     batch = loader.get_batch(50)
    #     if np.max(batch['annotation']) > 0.:
    #         print('hit!')


if __name__ == '__main__':
    test_loader(50)