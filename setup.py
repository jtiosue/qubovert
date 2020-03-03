#   Copyright 2020 Joseph T. Iosue
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


with open('README.rst') as f:
    README = f.read()

with open("requirements.txt") as f:
    REQUIREMENTS = [line.strip() for line in f if line.strip()]

# get __version__, __author__, etc.
with open("qubovert/_version.py") as f:
    exec(f.read())

# create the extension for the C file in qubovert.sim.src
extensions = [
    setuptools.Extension(
        name='qubovert.sim._simulate_quso',
        sources=['./qubovert/sim/_simulate_quso.pyx',
                 './qubovert/sim/src/_simulate_quso.c'],
        include_dirs=['./qubovert/sim/src/'],
        language='c',
    )
]

setuptools.setup(
    name="qubovert",
    version=__version__,
    author=__author__,
    author_email=__authoremail__,
    description=__description__,
    long_description=README,
    long_description_content_type='text/x-rst',
    url=__sourceurl__,
    license=__license__,
    packages=setuptools.find_packages(exclude=("tests", "docs")),
    ext_modules=extensions,
    test_suite="tests",
    setup_requires=REQUIREMENTS,
    install_requires=REQUIREMENTS,
    cmdclass={"build_ext": setuptools.command.build_ext.build_ext},
    zip_safe=False,
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    project_urls={
        "Source": __sourceurl__,
        "Docs": __docsurl__
    }
)
