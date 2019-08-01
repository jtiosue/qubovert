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

"""``partitioning`` contains many NP partitioning problems.

Here we have the QUBO/Ising conversions for common partitioning probems,
The conversions are based on [Lucas].

References
----------
.. [Lucas] Andrew Lucas. Ising formulations of many np problems. Frontiers in
Physics, 2:5, 2014.

"""

from ._number_partitioning import NumberPartitioning
from ._graph_partitioning import GraphPartitioning

__all__ = (
    "NumberPartitioning",
    "GraphPartitioning",
)

name = "partitioning"
