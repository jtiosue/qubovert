//   Copyright 2020 Joseph T. Iosue
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.

#ifndef simulate_quso_INCLUDED
#define simulate_quso_INCLUDED

void simulate_quso(
	int len_state, int *state, double *h,
	int *num_neighbors, int *neighbors, double *J,
	int len_Ts, double *Ts, int *num_updates,
	int in_order, int seed
);

#endif
