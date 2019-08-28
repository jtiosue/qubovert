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

"""``utils`` contains many utilities and helpers.

See ``__all__`` for a list of the utilities.

"""

# import order here is important!
from ._hash import *
from ._solve_bruteforce import *
from ._dict_arithmetic import *
from ._pubomatrix import *
from ._hisingmatrix import *
from ._qubomatrix import *
from ._isingmatrix import *
from ._conversions import *
from ._problem_parentclass import *
from ._bo_parentclass import *

from ._hash import __all__ as __all_hash__
from ._solve_bruteforce import __all__ as __all_solve_bruteforce__
from ._dict_arithmetic import __all__ as __all_dict_arithmetic__
from ._pubomatrix import __all__ as __all_pubomatrix__
from ._hisingmatrix import __all__ as __all_hisingmatrix__
from ._qubomatrix import __all__ as __all_qubomatrix__
from ._isingmatrix import __all__ as __all_isingmatrix__
from ._conversions import __all__ as __all_conversions__
from ._problem_parentclass import __all__ as __all_problem_parentclass__
from ._bo_parentclass import __all__ as __all_bo__


__all__ = (
    __all_hash__ +
    __all_solve_bruteforce__ +
    __all_dict_arithmetic__ +
    __all_pubomatrix__ +
    __all_hisingmatrix__ +
    __all_qubomatrix__ +
    __all_isingmatrix__ +
    __all_conversions__ +
    __all_problem_parentclass__ +
    __all_bo__
)


name = "utils"
