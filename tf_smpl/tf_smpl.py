import pickle
import tensorflow as tf
from .layers import AxisAngleToMatrix
from .layers import PoseSkeleton
from .layers import BlendSkinning
from .layers import VertexNormals

class SMPL(tf.keras.layers.Layer):
  def __init__(self, path, **kwargs):
    super(SMPL, self).__init__(**kwargs)
    with open(path, "rb") as __file:
      params = pickle.load(__file, encoding="latin1")

    with tf.name_scope("smpl"):
      self.v_template = tf.convert_to_tensor(
        value=params["v_template"],
        name="v_template",
        dtype=self.dtype
      )
      self.faces = tf.convert_to_tensor(
        value=params["f"],
        name="faces",
        dtype=tf.int32
      )
      self.shapedirs = tf.convert_to_tensor(
        value=params["shapedirs"].reshape([-1, params["shapedirs"].shape[-1]]).T,
        name="shapedirs",
        dtype=self.dtype
      )
      self.posedirs = tf.convert_to_tensor(
        value=params["posedirs"].reshape([-1, params["posedirs"].shape[-1]]).T,
        name="posedirs",
        dtype=self.dtype
      )
      self.J_regressor = tf.convert_to_tensor(
        value=params["J_regressor"].T.todense(),
        name="J_regressor",
        dtype=self.dtype
      )

      self.axis_to_matrix = AxisAngleToMatrix(name="axis_angle_to_matrix", dtype=self.dtype)
      self.pose_skeleton = PoseSkeleton(kintree_table=params["kintree_table"], dtype=self.dtype)
      self.blend_skinning = BlendSkinning(v_weights=params["weights"], dtype=self.dtype)
      self.vertex_normals = VertexNormals(dtype=self.dtype)

  def call(self, shapes, poses=None, trans=False):
    prefix_shape  = shapes.shape[:-1]
    tensor_dict = {
      "shapes": shapes,
      "poses": poses,
      "trans": trans
    }

    shapes = tf.reshape(shapes, [-1, 10])
    v_shaped = self.v_template + tf.reshape(
      tensor=tf.matmul(shapes, self.shapedirs),
      shape=[-1, self.v_template.shape[0], 3],
      name="v_ssd")

    if poses is None:
      if trans is not None:
        v_shaped += tf.reshape(trans, shape=(-1, 3))[:, tf.newaxis, :]
      return v_shaped

    poses = tf.reshape(poses, [-1, self.J_regressor.shape[1], 3])

    J_rotations = tf.convert_to_tensor(
      value=self.axis_to_matrix(poses),
      name="J_rotations",
      dtype=self.dtype
    )
    J_locations = tf.stack(
      values=[
        tf.matmul(v_shaped[:, :, 0], self.J_regressor),
        tf.matmul(v_shaped[:, :, 1], self.J_regressor),
        tf.matmul(v_shaped[:, :, 2], self.J_regressor),
      ],
      axis=2,
      name="J_locations"
    )
    lrotmin = tf.reshape(
      tensor=J_rotations[:, 1:, :, :] - tf.eye(3),
      shape=[-1, 9 * (self.J_regressor.shape[1] - 1)],
      name="lrotmin"
    )
    v_posed = v_shaped + tf.reshape(
      tensor=tf.matmul(lrotmin, self.posedirs),
      shape=[-1, self.v_template.shape[0], 3],
      name="v_psd"
    )
    J_transforms, J_locations = self.pose_skeleton(J_rotations, J_locations)
    v_body = self.blend_skinning(v_posed, J_transforms)

    if trans is not None:
      v_body += tf.reshape(trans, shape=(-1, 3))[:, tf.newaxis, :]
    
    v_body = tf.reshape(
      tensor=v_body,
      shape=prefix_shape + v_body.shape[1:],
      name="v_body"
    )
    tensor_dict["v_shaped"] = tf.reshape(
      tensor=v_shaped,
      shape=prefix_shape + v_shaped.shape[1:],
      name="v_shaped"
    )
    tensor_dict["v_posed"] = tf.reshape(
      tensor=v_posed,
      shape=prefix_shape + v_posed.shape[1:],
      name="v_posed"
    )
    tensor_dict["J_rotations"] = tf.reshape(
      tensor=J_rotations,
      shape=prefix_shape + J_rotations.shape[1:],
      name="J_rotations"
    )
    tensor_dict["J_locations"] = tf.reshape(
      tensor=J_locations,
      shape=prefix_shape + J_locations.shape[1:],
      name="J_rotations"
    )
    tensor_dict["J_transforms"] = tf.reshape(
      tensor=J_transforms,
      shape=prefix_shape + J_transforms.shape[1:],
      name="J_rotations"
    )

    return v_body, tensor_dict

  def normals(self, v_body):
    prefix_shape  = v_body.shape[:-2]
    v_body = tf.reshape(
      tensor=v_body,
      shape=[-1, v_body.shape[-2], 3],
      name="v_body"
    )
    normals = self.vertex_normals(v_body, self.faces)
    return tf.reshape(
      tensor=normals,
      shape=prefix_shape + v_body.shape[1:],
      name="v_normals"
    )
