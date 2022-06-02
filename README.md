# TensorFlow implementation of SMPL
This repository contains the code to easy use [SMPL](https://files.is.tue.mpg.de/black/papers/SMPL2015.pdf) with TensorFlow.

## Installation
```
$ pip install git+https://github.com/opendeeple/tf-smpl.git
```

## Download SMPL model
- Sign in into https://smpl.is.tue.mpg.de
- Download SMPL version 1.0.0 for Python 2.7 (10 shape PCs)
- Extract SMPL_python_v.1.0.0.zip

## Usage
```py
import tensorflow as tf
from tf_smpl import SMPL

smpl = SMPL("<smpl_model_extraced_folder>/basicModel_f_lbs_10_207_0_v1.0.0.pkl")
v_body, body_dict = smpl(
  shapes=tf.zeros(shape=[16, 3, 10]),
  poses=tf.zeros(shape=[16, 3, 72]),
  trans=tf.zeros(shape=[16, 3, 3])
)
```
