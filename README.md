# pencil_cifar100

We will implement neural network models for image classification problem with label noise and class imbalance.

The second assignment let you implement any neural network models for image classification on CIFAR100 dataset with noisy label.
Noisy label here means that some of the given labels are not correct (e.g., it is a dog image but labeled as a cat).

### Download the dataset

Go to [dataset page in our kaggle page for this challenge (CIFAR100-NoisyLabel)](https://www.kaggle.com/c/cifar100-image-classification-with-noisy-labels/data) to download the dataset. Copy (or move) the dataset into `./dataset` sub-directory.

### Files

make sure dataset file is well organized as below

```
cv-proj-pencil
├── PENCIL.py
├── resnet34.py
`── dataset
    │── dataset.py
    │── data
    │   │--cifar100_nl_test.csv
    │    `-- cifar100_nl.csv
    │
    │── cifar100_nl/img
        │-- test
        │   │-- RvmEkKNaAS.png
        │   │-- jsBXnkYbax.png
        │   │-- tSGoIoLbQX.png
        │   │-- ...
        `-- train
            │-- IGsDLlvWEG.png
            │-- gzDCjnjiBq.png
            │-- nGeAohKpVk.png
            │-- ...
```

Each line of the dataset's `csv` file follows the below format:
```
filename,classname
```

Train & validation for cifar100
```
python PENCIL.py
```

Test for cifar100 with best model
```
python PENCIL.py --evaluate --best True
```

reference : 
```
@inproceedings{PENCIL_CVPR_2019,
author = {Kun, Yi and Jianxin, Wu},
title = {{Probabilistic End-to-end Noise Correction for Learning with Noisy Labels}},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
year = {2019}
}
```
