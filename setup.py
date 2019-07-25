"""
Set up details for `pip install qubovert` or `pip install -e .` if installing
by source.
"""

import setuptools
from qubovert import __version__


with open('README.md') as f:
    README = f.read()

with open('LICENSE') as f:
    LICENSE = f.read()

with open("requirements.txt") as f:
    REQUIREMENTS = [line.strip() for line in f if line.strip()]


setuptools.setup(
    name="qubovert",
    version=__version__,
    author="Joseph Iosue",
    author_email="joe.iosue@yahoo.com",
    description="A package for converting common problems to QUBO form",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/jiosue/qubovert",
    license=LICENSE,
    packages=setuptools.find_packages(exclude=("tests", "docs")),
    test_suite="tests",
    install_requires=REQUIREMENTS,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
