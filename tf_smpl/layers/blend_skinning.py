import tensorflow as tf
from tensorflow import keras

class BlendSkinning(tf.keras.layers.Layer):
  def __init__(self, v_weights, **kwargs):
    super(BlendSkinning, self).__init__(**kwargs)
    self.v_weights = tf.Variable(
      initial_value=v_weights,
      name="v_weights",
      dtype=self.dtype
    )

  def call(self, vertices, J_rotations):
    prefix_shape = vertices.shape[:-2]
    vertices = tf.reshape(vertices, shape=[-1, vertices.shape[-2], 3])
    J_rotations = tf.reshape(J_rotations, shape=[-1, *J_rotations.shape[-3:]])

    batch_size = tf.shape(vertices)[0]
    num_joints = self.v_weights.shape[-1]
    num_vertices = vertices.shape[-2]

    W = self.v_weights
    if len(self.v_weights.shape.as_list()) < len(vertices.shape.as_list()):
      W = tf.tile(tf.convert_to_tensor(self.v_weights), [batch_size, 1])
      W = tf.reshape(W, [batch_size, self.v_weights.shape[-2], num_joints])

    A = tf.reshape(J_rotations, (-1, num_joints, 16))
    T = tf.matmul(W, A)
    T = tf.reshape(T, (-1, num_vertices, 4, 4))

    ones = tf.ones([batch_size, num_vertices, 1])
    vertices_homo = tf.concat([vertices, ones], axis=2)
    skinned_homo = tf.matmul(T, tf.expand_dims(vertices_homo, -1))
    skinned_vertices = skinned_homo[:, :, :3, 0]

    return tf.reshape(
      tensor=skinned_vertices,
      shape=prefix_shape + skinned_vertices.shape[1:],
      name="v_skinned"
    )
