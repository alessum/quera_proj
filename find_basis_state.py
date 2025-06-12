# Used to find specific iniitial computational basis states
import numpy as np
import functools as ft

up, down = np.array([1, 0]), np.array([0, 1])
list_states = [down]*13
list_states[0] = up
list_states[4] = up
list_states[8] = up
list_states[12] = up

full_state = ft.reduce(np.kron, list_states)

# finding nonzero index
nonzero_indices = np.nonzero(full_state)[0]
print("Non-zero indices:", nonzero_indices)