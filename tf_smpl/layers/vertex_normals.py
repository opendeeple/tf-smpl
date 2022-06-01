import tensorflow as tf

class FaceNormals(tf.keras.layers.Layer):
  def __init__(self, normalize=True, **kwargs):
    super(FaceNormals, self).__init__(**kwargs)
    self.normalize = normalize

  def call(self, vertices, faces):
    v = vertices
    f = faces

    if v.shape.ndims == (f.shape.ndims + 1):
      f = tf.tile([f], [tf.shape(v)[0], 1, 1])   

    triangles = tf.gather(v, f, axis=-2, batch_dims=v.shape.ndims - 2) 

    v0, v1, v2 = tf.unstack(triangles, axis=-2)
    e1 = v0 - v1
    e2 = v2 - v1
    face_normals = tf.linalg.cross(e2, e1) 

    if self.normalize:
      face_normals = tf.math.l2_normalize(face_normals, axis=-1)
    return face_normals

class VertexNormals(tf.keras.layers.Layer):
  def __init__(self, normalize=True, **kwargs):
    super(VertexNormals, self).__init__(**kwargs)
    self.normalize = normalize
    self.face_normals = FaceNormals(normalize=False, dtype=self.dtype)

  def call(self, vertices, faces):
    batch_size = tf.shape(vertices)[0]

    if not tf.is_tensor(faces):
      faces = tf.convert_to_tensor(faces, dtype=tf.int32)

    faces_flat = tf.reshape(faces, [-1])
    faces_tiled = tf.tile(faces_flat, [batch_size])
    faces = tf.reshape(faces_tiled, [batch_size] + faces.shape.as_list())

    shape_faces = faces.shape.as_list()
    mesh_face_normals = self.face_normals(vertices, faces)

    outer_indices = tf.range(batch_size, dtype=tf.int32)
    outer_indices = tf.expand_dims(outer_indices, axis=-1)
    outer_indices = tf.expand_dims(outer_indices, axis=-1)
    outer_indices = tf.tile(outer_indices, [1] * len(shape_faces[:-2]) + [tf.shape(input=faces)[-2]] + [1])

    vertex_normals = tf.zeros_like(vertices)
    for i in range(shape_faces[-1]):
      scatter_indices = tf.concat(
        [outer_indices, faces[..., i:i + 1]], axis=-1)

      vertex_normals = tf.compat.v1.tensor_scatter_add(
        vertex_normals, scatter_indices, mesh_face_normals)

    if self.normalize:
      vertex_normals = tf.math.l2_normalize(vertex_normals, axis=-1)

    return vertex_normals
