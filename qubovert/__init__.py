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

"""A module for converting problems into QUBO/QUSO form.

QUBO stands for Quadratic Unconstrained Boolean Optimization. QUBO problems
have a one-to-one mapping to classical QUSO problems, and most optimization
problems are formatted in QUBO form when the solver is a quantum computer.
See ``qubovert.__all__`` for useful functionality, ``qubovert.problems__all__``
for problems defined, and ``qubovert.utils.__all__`` for some utility
functions, ``qubovert.sat.__all__`` for the satisfiability library, and
``qubovert.sim.__all__`` for the simulation and annealing library.

"""

from ._version import *

from . import utils

from ._qubo import *
from ._quso import *
from ._pubo import *
from ._puso import *
from ._pcbo import *
from ._pcso import *

from ._qubo import __all__ as __all_qubo__
from ._quso import __all__ as __all_quso__
from ._pubo import __all__ as __all_pubo__
from ._puso import __all__ as __all_puso__
from ._pcbo import __all__ as __all_pcbo__
from ._pcso import __all__ as __all_pcso__

__all__ = (
    __all_qubo__ +
    __all_quso__ +
    __all_pubo__ +
    __all_puso__ +
    __all_pcbo__ +
    __all_pcso__
)

from . import sat
from . import sim
from . import problems


del __all_qubo__, __all_quso__
del __all_pubo__, __all_puso__
del __all_pcbo__, __all_pcso__

BOOLEAN_MODELS = QUBO, PUBO, PCBO, utils.QUBOMatrix, utils.PUBOMatrix
SPIN_MODELS = QUSO, PUSO, PCSO, utils.QUSOMatrix, utils.PUSOMatrix


name = "qubovert"
