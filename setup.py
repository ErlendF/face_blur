from setuptools import setup, find_packages

setup(
    name='ml_utils',
    version='0.1',
    description='ML utilities',
    author='Erlend Fonnes',
    packages=find_packages(),
    install_requires=["numpy", "opencv-python",
                      "bisect", "copy", "matplotlib.pyplot", "face_recognition"]
)
