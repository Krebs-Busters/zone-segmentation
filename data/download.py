from tqdm import tqdm
import urllib.request
from pathlib import Path
import zipfile
import pdb

data_location = 'https://zenodo.org/record/6624726'

files = ["picai_public_images_fold{}.zip".format(i) for i in range(0,4)]
picai_url = [(data_location + '/files/' + file, 'full/' + file) for file in files]


def load_and_store(file_list):
    for online_path, local_path in tqdm(file_list, desc='downloading'):
        urllib.request.urlretrieve(online_path, local_path)

print("dowloading... this will take a while. Please be patient.")
Path("full/").mkdir(exist_ok=True)
load_and_store(picai_url)
print("download ok")

print("Extracting...")
def extract_zip_in_folder(file_list):
    for file in tqdm(file_list, desc='extracting'):
        with zipfile.ZipFile(file, 'r') as zip:
            folder = Path('full/' + file.split('/')[-1][:-4])
            folder.mkdir(exist_ok=True)
            zip.extractall(path=folder)


local_zips = list(purl[1] for purl in picai_url)
extract_zip_in_folder(local_zips)