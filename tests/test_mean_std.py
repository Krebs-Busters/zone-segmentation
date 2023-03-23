import numpy as np

from src.medseg.data_loader import PicaiLoader

def test_mean_std():
    loader = PicaiLoader()

    # look for people with cancer.
    images = []
    for pos, key in enumerate(loader.patient_keys):
        image = loader.get_record(key)['images']
        images.append(image.astype(np.float32).flatten())
        print(pos, key)
    data_set = np.concatenate(images, -1)
    del images
    mean = np.mean(data_set)
    std = np.std(data_set)
    print(mean, std)
    np.allclose([206.12558, 164.74423], [mean, std], atol=1e-6)

