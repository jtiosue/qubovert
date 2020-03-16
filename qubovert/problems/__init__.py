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

"""``problems`` contains many problems converted to QUBO/QUSO form.

This module contains many all the problems that we convert to QUBO/QUSO form.
We organize all the problems into categories, but we import them globally for
user use. See the ``__all__`` value for all the problems imported. Each of the
problems inherits from the ``Problem`` parent class, see
``qubovert.problems.Problem``.

"""

from ._problem_parentclass import *  # don't add to __all__!

from .np import *
from .benchmarking import *

from .np import __all__ as __all_np__
from .benchmarking import __all__ as __all_benchmarking__

__all__ = (
    __all_np__ + __all_benchmarking__
)

del __all_np__, __all_benchmarking__

name = "problems"
