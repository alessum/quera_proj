# Used to find specific iniitial computational basis states
import numpy as np
import functools as ft

up, down = np.array([1, 0]), np.array([0, 1])
list_states = [up]*16
list_states[0] = down
list_states[3] = down
list_states[12] = down
list_states[15] = down

full_state = ft.reduce(np.kron, list_states)

# finding nonzero index
nonzero_indices = np.nonzero(full_state)[0]
print("Non-zero indices:", nonzero_indices)