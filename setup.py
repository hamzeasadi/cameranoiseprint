from setuptools import setup, find_packages


VERSION = 0.1
DESCRIPTION = 'camera source identification based on residual learning'
LONG_DESCRIPTION = 'A python package to manage data and training pipeline of a video source identification'

"""
we need to run following command to use the package
python setup.py sdist bdist_wheel

4. Install the local package

At the project's root directory, run the terminal command below to install the package in the current working directory (.) in editable mode (-e).

pip install -e .
"""
setup(
    name='cameranoiseprint',
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author='hamzeh',
    packages=find_packages()
)