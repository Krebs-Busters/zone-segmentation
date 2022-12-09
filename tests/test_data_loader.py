from src.medseg.data_loader import DataLoader


def test_loader():
    loader = DataLoader()
    # rec = loader.get_record()
    batch = loader.get_batch(50)