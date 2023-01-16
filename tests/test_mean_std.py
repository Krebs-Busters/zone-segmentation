from src.medseg.data_loader import PicaiLoader

def test_mean_std():
    picai = PicaiLoader()
    img_lst = []
    for key in picai.patient_keys:
        img_lst.append(picai.get_record[key]['images']['t2w'])