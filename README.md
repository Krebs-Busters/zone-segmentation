Cancer-segmentation
-------------------

### Getting started
Start by navigating into the data folder and run `python download.py` to set up the training data.

### Requirements
All the libraries required to run this project are specified in requirements.txt

### Training & Validation
For training, run `python train.py`.
In case of training for the first time, change the variable `load_new` to True.
With label new, dataset is divided into batches and saved as pickled, saving the precious batching time for future training.
Once the training is done, weights are saved as pickled file in `./weights` path.'

### Model
A 3D U-Net model is modelled in `train.py` as specified in this [paper](https://arxiv.org/pdf/1505.04597.pdf).

### To test the model run
```bash
python src/medseg/sample.py
```

### Skyra_Mannheim
- Drei Teile zusammenfügen.
- 