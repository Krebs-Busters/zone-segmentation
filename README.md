Cancer-segmentation
-------------------

### Getting started
Start by navigating into the data folder and run `python download.py` to set up the training data.
To clone via ssh configure your local system for ssh-access as described in the [github-docs](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account).

### Using the model
To clone this repository, type:
``` bash
git clone git@github.com:Krebs-Busters/zone-segmentation.git
```
If outgoing SSH is disabled in your network, download the repository using your web browser, using the `download ZIP` button.
After extracting the zip file set up the Python environment by running:
``` bash
pip install -r requirements.txt
```
TODO fertigschreiben.



### Training & Validation
For training, run `python train.py`.
In case of training for the first time, change the variable `load_new` to True.
Once the training is done, weights are saved as pickled file in `./weights` path.'

### Model
A 3D U-Net model is modelled in `train.py` as specified in this [paper](https://arxiv.org/pdf/1505.04597.pdf).

### To test the model, run
``` bash
python src/medseg/sample.py
```

### Citation
Should you use this work in an academic context please cite:


``` 
@inproceedings{meyer2019towards,
  title={Towards patient-individual PI-Rads v2 sector map: CNN for automatic segmentation of prostatic zones from T2-weighted MRI},
  author={Meyer, Anneke and Rakr, Marko and Schindele, Daniel and Blaschke, Simon and Schostak, Martin and Fedorov, Andriy and Hansen, Christian},
  booktitle={2019 IEEE 16th International Symposium on Biomedical Imaging (ISBI 2019)},
  pages={696--700},
  year={2019},
  organization={IEEE}
}
```

```
@software{wolter2024stability,
  title={On the Stability of Neural Segmentation in Radiology},
  author={Wolter, Moritz and Wichtmann, Barbara},
  url = {https://github.com/Krebs-Busters/zone-segmentation}
  year={2024},
}
```
