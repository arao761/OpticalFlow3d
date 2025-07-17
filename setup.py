from setuptools import setup, find_packages
from os import path

with open("README.md", "r") as file:
    long_description = file.read()

_dir = path.dirname(__file__)

dependencies = [
    "numpy>=1.17.0",
    "scikit-image>=0.17.1",
    "scipy>=1.6.3",
    "tqdm>=4.50.0",
    "torch>=1.13.0"
]

setup(
    name='opticalflow3d',
    version="0.3.2",
    description='PyTorch optimized implementation of 3D optical flow algorithms such as Farneback two frame motion estimation and Lucas Kanade dense optical flow algorithms',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=dependencies,
    author='Xianbin Yong',
    author_email='xianbin.yong13@sps.nus.edu.sg',
    url='https://gitlab.com/xianbin.yong13/opticalflow3d',

    packages=find_packages(),
    license="GPLv3",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering",
    ],
    python_requires='>=3.7',
    project_urls={
        'Research group': 'https://ctlimlab.org/',
        'Source': 'https://gitlab.com/xianbin.yong13/opticalflow3d',
    },
)
