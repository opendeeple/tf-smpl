# TensorFlow implementation of SMPL (A Skinned Multi-Person Linear Model)
This repository contains the code to easy use [SMPL](https://files.is.tue.mpg.de/black/papers/SMPL2015.pdf) with TensorFlow.

## Installation
```
$ pip install git+https://github.com/opendeeple/tf-smpl.git
```

## Download SMPL model
- Sign in into https://smpl.is.tue.mpg.de
- Download SMPL version 1.0.0 for Python 2.7 (10 shape PCs)
- Extract SMPL_python_v.1.0.0.zip

## Run
```
$ smpl --config configs/example.conf --motion motions\*.npz
```

## Usage
```py
import tensorflow as tf
from tf_smpl import SMPL

smpl = SMPL("<smpl_model_extraced_folder>/basicModel_f_lbs_10_207_0_v1.0.0.pkl")
# calculate SMPL vertices
batch_size = 16
v_body = smpl(
  shapes=tf.zeros(shape=[batch_size, 10]), # sample shapes (betas)
  poses=tf.zeros(shape=[batch_size, 72]), # sample poses
  trans=tf.zeros(shape=[batch_size, 3]) # sample trans
)
# calculate SMPL vertices for sequences
sequence_size = 3
v_body = smpl(
  shapes=tf.zeros(shape=[batch_size, sequence_size, 10]), # sample shapes (betas)
  poses=tf.zeros(shape=[batch_size, sequence_size, 72]), # sample poses
  trans=tf.zeros(shape=[batch_size, sequence_size, 3]) # sample trans
)
```
Calculate SMPL vertices with get middle variables

List of midle variables
- v_shaped
- v_posed
- J_rotations
- J_locations
- J_transforms
```py
v_body, body_dict = smpl(
  ...,
  includes=["v_shaped", "J_locations"] # for get middle variables
)
print(body_dict) # display body_dict:middle variables
```
Calculate SMPL normals
```py
v_normals = smpl.normals(v_body)
```
Calculate neighbours of SMPL with Outfit
```py
neighbours = smpl.neighbours(v_body, v_outfit)
```