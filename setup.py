import setuptools

setuptools.setup(
  name="tf_smpl",
  version="1.0.0",
  description="TensorFlow implementation of SMPL",
  url="https://github.com/opendeeple/tf-smpl",
  author="Firdavs Beknazarov",
  author_email="opendeeple@gmail.com",
  packages = setuptools.find_packages(),
  install_requires = ["tensorflow"],
)
