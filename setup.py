#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="src",
    version="1.0.0",
    description="Monocular 3D Object Detection",
    author="Didi Ruhyadi",
    author_email="ruhyadi.dr@gmail.com",
    url="https://github.com/ruhyadi/yolo3d",
    install_requires=["lightning", "hydra-core"],
    packages=find_packages(),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "train_command = src.train:main",
            "eval_command = src.eval:main",
        ]
    },
)
