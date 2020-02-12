#   Copyright 2019 Joseph T. Iosue
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""setup.py.

Set up details for ``pip install qubovert`` or ``pip install -e .`` if
installing by source.

"""

import setuptools
from qubovert import __version__, __name__


with open('README.rst') as f:
    README = f.read()

with open("requirements.txt") as f:
    REQUIREMENTS = [line.strip() for line in f if line.strip()]


setuptools.setup(
    name=__name__,
    version=__version__,
    author="Joseph T. Iosue",
    author_email="joe.iosue@yahoo.com",
    description="A package for converting problems to boolean and spin form",
    long_description=README,
    long_description_content_type='text/x-rst',
    url="https://github.com/jiosue/qubovert",
    license="Apache Software License 2.0",
    packages=setuptools.find_packages(exclude=("tests", "docs")),
    test_suite="tests",
    install_requires=REQUIREMENTS,
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    project_urls={
        "Source": "https://github.com/jiosue/qubovert",
        "Docs": "https://qubovert.readthedocs.io"
    }
)
