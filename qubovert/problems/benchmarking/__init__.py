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

"""``benchmarking`` contains many benchmarking example problems.

This module contains many benchmarking problems. We import all the problems
globally for user use. See the ``__all__`` value for all the problems imported.

"""

from ._alternating_sectors_chain import *

from ._alternating_sectors_chain import __all__ as __all_asc__

__all__ = (
    __all_asc__
)

del __all_asc__

name = "benchmarking"
