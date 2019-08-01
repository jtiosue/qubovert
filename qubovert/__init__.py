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

"""qubovert is a module for converting common problems into QUBO/Ising form.
QUBO stands for Quadratic Unconstrained Binary Optimization. QUBO problems
have a one-to-one mapping to classical Ising problems, and most optimization
problems are formatted in QUBO form when the solver is a quantum computer.
See `qubovert.__all__` for all the problems defined, and
`qubovert.utils.__all__` for some utilities used.
"""

from ._version import __version__
from . import utils
from .problems import *

from .problems import __all__ as __all_problems__

# if someone does `from qubovert import *`, import all of the problems, but
# not utils.
__all__ = __all_problems__

name = "qubovert"
