#!/usr/bin/python3
from setuptools import setup, find_packages

requirements = [
    'numpy',
    'matplotlib',
    'imageio-ffmpeg',
    'scipy',
    'rich'
]

setup(
    name='cart-pole',
    description='Simulation and visualization for the cart-pole system.',
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    packages=find_packages(),
    author="Martin Ansteensen",
    python_requires='>=3.8',
    install_requires=requirements
)
