import os
import shutil
import pickle
import pathlib
from tcia_utils import nbia

if not os.path.exists("./data/"):
    os.makedirs("./data/")


# download the segmentations
download_from = 'https://isgwww.cs.uni-magdeburg.de/cas/isbi2019/Data.zip'

import requests, zipfile, io
r = requests.get(download_from)
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall("./data/")
shutil.move("./data/GT Export", "./data/gtexport")

print('annotation download complete.')

annotations = list(
    pathlib.Path("./data/gtexport/Train/").glob("*/*.nrrd"))
annotations_test = list(
    pathlib.Path("./data/gtexport/Test/").glob("*/*.nrrd"))
annotations += annotations_test

annotation_dict = {}
for annotation_path in annotations:
    patient_id = str(annotation_path).split('/')[-2]
    annotation_dict[patient_id] = annotation_path

# download the images
nbia.getCollections()
index = nbia.getSeries("ProstateX")

with open('./data/scan_index.pkl', 'wb') as f:
    pickle.dump(index, f)

patient_series_dict = {}
for entry in index:
    id = entry['PatientID']
    seriesuid = entry['SeriesInstanceUID']
    try:
        protocol = entry['ProtocolName']
        if id not in patient_series_dict.keys():
            patient_series_dict[id] = {}
        
        patient_series_dict[id][protocol] = seriesuid
    except:
        pass
        # print(f"suid {seriesuid}, Id: {id} missing protocol")

# verify data
patients_ok = []
interesting_protocols = ['t2_tse_tra', 't2_tse_sag', 't2_tse_cor']
for key, patient in patient_series_dict.items():
    test = []
    for protocol_key in interesting_protocols:
        test.append(protocol_key in patient.keys())
    if all(test):
        patients_ok.append(key)
print(f"found {len(patients_ok)} complete scans in metadata.")

# check overlap
for test_key in annotation_dict:
    assert test_key in patients_ok

to_download = []
for anno_key in annotation_dict:
    suids = [(patient_series_dict[anno_key][prot]) for prot in ['t2_tse_tra', 't2_tse_sag', 't2_tse_cor']]
    to_download.extend(suids)

index = list(filter(lambda fl: fl['SeriesInstanceUID'] in to_download, index))

nbia.downloadSeries(index)
shutil.move("tciaDownload", "./data/tciaDownload")

print("raw data download done.")
