Cancer-segmentation
-------------------

### Getting started
To clone via ssh configure your local system for ssh-access as described in the [github-docs](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account).

To clone this repository, type:
``` bash
git clone git@github.com:Krebs-Busters/zone-segmentation.git
```
If outgoing SSH is disabled in your network, download the repository using your web browser, using the `download ZIP` button.
After extracting the zip file set up the Python environment by running:
``` bash
pip install -r requirements.txt
```

#### Automatic setup of the Prostate-X data for training and testing.
Navigate into the data folder and run `python download.py` to set up the Prostate-X data-set.


### Training & Validation
For training, run
``` bash
PYTHONPATH=. python scripts/train_prostate_X.py
```
.
In case of training for the first time, change the variable `load_new` to True.
Once the training is done, weights are saved as a pickled file in `./weights`.

### Model
`src/networks.py` implements a 3D U-Net flax-model as specified by [Meyer et al.](https://arxiv.org/pdf/1505.04597.pdf).

### To test the model, run
``` bash
PYTHONPATH=. python scripts/sample_prostate_X.py
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
