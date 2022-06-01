import tensorflow as tf
from tensorflow import keras

class PoseSkeleton(keras.layers.Layer):
  def __init__(self, kintree_table, **kwargs):
    super(PoseSkeleton, self).__init__(**kwargs)
    self.parents = tf.convert_to_tensor(
      value=kintree_table[0],
      name="parents",
      dtype=tf.int32
    )

  def call(self, joint_rotations, joint_positions):
    batch_size = tf.shape(joint_rotations)[0]
    num_joints = len(self.parents)

    def make_affine(rotation, translation, name=None):
      rotation_homo = tf.pad(rotation, [[0, 0], [0, 1], [0, 0]])
      translation_homo = tf.concat([translation, tf.ones([batch_size, 1, 1])], 1)
      affine_transform = tf.concat([rotation_homo, translation_homo], 2)
      return affine_transform

    joint_positions = tf.expand_dims(joint_positions, axis=-1)      
    root_rotation = joint_rotations[:, 0, :, :]
    root_transform = make_affine(root_rotation, joint_positions[:, 0])

    transforms = [root_transform]
    for joint, parent in enumerate(self.parents[1:], start=1):
      position = joint_positions[:, joint] - joint_positions[:, parent]
      transform_local = make_affine(joint_rotations[:, joint], position)
      transform_global = tf.matmul(transforms[parent], transform_local)
      transforms.append(transform_global)
    transforms = tf.stack(transforms, axis=1) 

    joint_positions_posed = transforms[:, :, :3, 3]

    zeros = tf.zeros([batch_size, num_joints, 1, 1])
    joint_rest_positions = tf.concat([joint_positions, zeros], axis=2)
    init_bone = tf.matmul(transforms, joint_rest_positions)
    init_bone = tf.pad(init_bone, [[0, 0], [0, 0], [0, 0], [3, 0]])
    joint_transforms = transforms - init_bone

    return joint_transforms, joint_positions_posed
