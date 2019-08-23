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
from ._dict_arithmetic import DictArithmetic
from ._qubo_matrix import QUBOMatrix, IsingCoupling, IsingField
from ._conversions import (
    qubo_to_ising, ising_to_qubo,
    matrix_to_qubo, qubo_to_matrix,
    binary_to_spin, spin_to_binary,
    decimal_to_binary, decimal_to_spin
)
from ._solve_bruteforce import (
    solve_qubo_bruteforce, solve_ising_bruteforce, qubo_value, ising_value
)
from ._problem_parentclass import Problem
from ._bo_parentclass import BO


__all__ = (
    "QUBOMatrix",
    "IsingCoupling",
    "IsingField",
    "qubo_to_ising",
    "ising_to_qubo",
    "matrix_to_qubo",
    "qubo_to_matrix",
    "binary_to_spin",
    "spin_to_binary",
    "decimal_to_binary",
    "decimal_to_spin",
    "solve_qubo_bruteforce",
    "solve_ising_bruteforce",
    "qubo_value",
    "ising_value",
    "Problem",
    "DictArithmetic",
    "BO",
)


name = "utils"
