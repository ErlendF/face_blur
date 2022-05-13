from setuptools import setup, find_packages

setup(
    name='blur_face',
    version='0.1',
    description='ML utilities',
    author='Erlend Fonnes',
    packages=find_packages(),
    install_requires=["numpy", "opencv-python", "matplotlib", "pandas",
                      "sympy", "nose", "scipy", "deepface", "keras", "tensorflow>=2.1"]
)
