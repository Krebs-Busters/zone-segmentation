import numpy as np

from src.medseg.data_loader import PicaiLoader

def test_mean_std():
    loader = PicaiLoader()

    # look for people with cancer.
    images = []
    for pos, key in enumerate(loader.patient_keys):
        image = loader.get_record(key)['images']
        images.append(image.astype(np.float32))
        print(pos, key)
    data_set = np.stack(images)
    del images
    mean = np.mean(data_set)
    std = np.std(data_set)
    print(mean, std)
    np.allclose([3.4925714, 31.330494], [mean, std])

