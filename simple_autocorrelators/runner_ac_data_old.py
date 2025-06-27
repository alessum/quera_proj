import os, io, imageio
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from hamiltonian_wrapper import RydbergLatticeSystem, plt, find_random_basis_state_j

# ─── 1) DEFINE A SIMPLE 2D LATTICE ──────────────────────────────────────
Lx, Ly = 3, 3                    # 3×3 grid → 9 sites
spacing = 5.93
positions = [(i * spacing, j * spacing) for i in range(Lx) for j in range(Ly)]
N_sites = len(positions)
#mid_site = N_sites // 2  # center site in the grid, works for odd L
mid_site = 4  # starting from the first site, can be changed to any other site
N_trials = 100
T_total, T_steps = 4.0, 400

h_arr = np.zeros(N_sites)  # no local fields
Delta_G, Delta_L, C6 = 125, 0.0, 5.42*(10**6)

def Omega_t(t):
    return 15.8 + 0.0*t

def phi_t(t):
    return 0.0 + 0.0*t

# 2) INSTANTIATE THE SYSTEM
system = RydbergLatticeSystem(
    positions=positions,
    h=h_arr,
    Delta_G=Delta_G,
    Delta_L=Delta_L,
    C6=C6,
    Omega_t=Omega_t,
    phi_t=phi_t,
)

# 3) plot the lattice geometry
# fig, ax = system.plot_lattice()
# plt.show()

# 4) BUILD THE TIME‐DEPENDENT HAMILTONIAN
H = system.build_hamiltonian()

for trial in tqdm(range(N_trials), desc="Running trials"):
    # 5) DEFINE INITIAL STATE |ψ₀⟩ = all spins ↓
    psi0 = np.zeros(2**system.Ns, dtype=np.complex128)
    state_ind = find_random_basis_state_j(N_sites, mid_site)
    psi0[state_ind] = 1.0

    # 6) TIME GRID
    T_steps = 200
    times = np.linspace(0.0, T_total, T_steps)

    # 7) COMPUTE ON‐SITE ZZ AUTOCORRELATOR FOR MID SITE
    corr = system.compute_zz_autocorrelator(psi0, times, sites=[mid_site])
    corr_abs = np.real(corr)          # shape = (N_sites, T_steps). # removed the abs and replaced with real part

    # 9) Saving data to csv file
    csv_filename = "ac_results/correlator_data_Lx{}_Ly{}_trial{}.csv".format(Lx, Ly, trial)
    np.savetxt(csv_filename, corr_abs, delimiter=",")