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

"""``sat`` is a module for converting SAT problems into QUBO/QUSO form.

``sat`` is a library of ``qubovert`` for converting satisfiability problems
into PUBOs (see ``help(qubovert.PUBO)``).

"""

from ._satisfiability import *

from ._satisfiability import __all__ as __all_sat__

__all__ = (
    __all_sat__
)

del __all_sat__

name = "sat"
