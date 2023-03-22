import os
import sys
import argparse
import numpy as np
from glob import glob
import tensorflow as tf
import configparser
from tqdm import tqdm
from struct import pack, unpack
from scipy.spatial.transform import Rotation as R

# python tf_smpl\bin.py --config assets\configs\example.conf --motion assets\poses\**\**\*.npz

if __name__ == "__main__" and __package__ is None:
  sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
  import tf_smpl.bin
  __package__ = "tf_smpl.bin"

from tf_smpl import SMPL

def args_parser():
	parser = argparse.ArgumentParser(description="Poses data to pc2 converter script using SMPL")
	parser.add_argument('--config', help="Config file path", required=True)
	parser.add_argument('--motion', help="Motion file path", required=True)
	args = parser.parse_args(sys.argv[1:])
	config = configparser.ConfigParser()
	config.read(args.config)
	args.config = config
	return args

def separate_arms(poses, angle=20, left_arm=17, right_arm=16):
  num_joints = poses.shape[-1] //3

  poses = poses.reshape((-1, num_joints, 3))
  rot = R.from_euler('z', -angle, degrees=True)
  poses[:, left_arm] = (rot * R.from_rotvec(poses[:, left_arm])).as_rotvec()
  rot = R.from_euler('z', angle, degrees=True)
  poses[:, right_arm] = (rot * R.from_rotvec(poses[:, right_arm])).as_rotvec()

  poses[:, 23] *= 0.1
  poses[:, 22] *= 0.1

  return poses.reshape((poses.shape[0], -1))

def load_motion(path):
  motion = np.load(path, mmap_mode='r')
  reduce_factor = int(motion["mocap_framerate"] // 30)
  pose = motion["poses"][::reduce_factor, :72]
  trans = motion["trans"][::reduce_factor, :]
  swap_rotation = R.from_euler('zx', [-90, 270], degrees=True)
  root_rot = R.from_rotvec(pose[:, :3])
  pose[:, :3] = (swap_rotation * root_rot).as_rotvec()
  trans = swap_rotation.apply(trans)
  return pose.astype(np.float32), trans.astype(np.float32)

def save_pc2(path, V, float16=False):
  if float16: V = V.astype(np.float16)
  else: V = V.astype(np.float32)
  with open(path, 'wb') as __file:
    header_format='<12siiffi'
    header_str = pack(header_format, b'POINTCACHE2\0', 1, V.shape[1], 0, 1, V.shape[0])
    __file.write(header_str)
    __file.write(V.tobytes())

def save_pc2_frames(path, V, float16=False):
  if os.path.isfile(path):
    if float16: V = V.astype(np.float16)
    else: V = V.astype(np.float32)
    with open(path, 'rb+') as __file:
      __file.seek(16)
      nPoints = unpack('<i', __file.read(4))[0]
      assert len(V.shape) == 3 and V.shape[1] == nPoints, 'Inconsistent dimensions: ' + str(V.shape) + ' and should be (-1,' + str(nPoints) + ',3)'
      __file.seek(28)
      nSamples = unpack('<i', __file.read(4))[0]
      nSamples += V.shape[0]
      __file.seek(28)
      __file.write(pack('i', nSamples))
      __file.seek(0, 2)
      __file.write(V.tobytes())
  else: save_pc2(path, V, float16)

def main():
	args = args_parser()
	data = glob(args.motion)
	for item_path in tqdm(data):
		poses, trans = load_motion(item_path)
		betas = np.array([list(map(float, args.config.get("smpl", "shape").split())) for _ in range(poses.shape[0])])
		smpl = SMPL(args.config.get("smpl", "path"))
		poses = tf.convert_to_tensor(value=poses, name="poses", dtype=tf.float32)
		betas = tf.convert_to_tensor(value=betas, name="betas", dtype=tf.float32)
		trans = tf.convert_to_tensor(value=trans, name="trans", dtype=tf.float32)
		v_body = smpl(shapes=betas, poses=poses, trans=trans)
		save_path = item_path[:-4] + ".pc2"
		if os.path.isfile(save_path):
			os.remove(save_path)
		save_pc2_frames(save_path, v_body.numpy())

if __name__ == "__main__":
	main()
