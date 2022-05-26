from setuptools import setup, find_packages

setup(
    name='face_blur',
    version='0.1',
    description='ML utilities',
    author='Erlend Fonnes',
    packages=find_packages(),
    install_requires=["numpy", "opencv-python",
                      "scipy", "deepface", "keras", "tensorflow>=2.1"]
)
