Cancer-segmentation
-------------------

### Getting started
Start by navigating into the data folder and run `python download.py` to download the [PI-CAI dataset](https://zenodo.org/record/6624726).
This might take some time to download and unzip. 
Don't foget to clone the annotation repo via `git clone https://github.com/DIAGNijmegen/picai_labels`.

### Requirements
All the libraries required to run this project is specified in requirements.txt

### Data-loader
Run,
```bash
./scripts/profile_data_loader.sh
```
to profile the data loader.

### Training & Validation
For training, run `python train.py`.
In case of training for the first time, change the variable `load_new` to True.
With label new, dataset is divided into batches and saved as pickled, saving the precious batching time for future training.
Once the training is done, weights are saved as pickled file in `./weights` path.'

### Model
A 3D U-Net model is specified in `train.py` as sepcified in this [paper](https://arxiv.org/pdf/1505.04597.pdf).