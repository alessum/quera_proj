import os, io, imageio
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from hamiltonian_wrapper import RydbergLatticeSystem, plt, find_random_basis_state_j
from math import pi
import random

# ─── 1) DEFINE A SIMPLE 2D LATTICE ──────────────────────────────────────
Lx, Ly = 3, 3                
spacing = 5.93
positions = [(i * spacing, j * spacing) for i in range(Lx) for j in range(Ly)]
N_sites = len(positions)
mid_site = N_sites // 2  # center site in the grid, works for odd Lx and Ly
print(mid_site)

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

# 5) DEFINE INITIAL STATE |ψ₀⟩ = random state with mid-site spin in state up: |↑⟩
psi0 = np.zeros(2**system.Ns, dtype=np.complex128)
state_ind = find_random_basis_state_j(N_sites, mid_site)
psi0[state_ind] = 1.0

# 6) TIME GRID
T_steps = 101
times = np.linspace(0.0, 4.0, T_steps)

# 7) COMPUTE THE TIME‐EVOLUTION OF THE SPIN‐Z EXPECTATION VALUE FOR ALL SITES
sz_evo = system.compute_sz_ev(psi0, times)
sz_evo = np.real(sz_evo)  # ensure we have real values
print("Shape of sz expectation value array:", sz_evo.shape)  # → (N_sites, T_steps)

# 8) PREPARE DATA FOR GIF: absolute value + linear color scale
vmin, vmax = sz_evo.min(), sz_evo.max()

# 9) BUILD AND SAVE A GIF OF THE TIME‐EVOLUTION, WITH A “TIME BAR” AT THE BOTTOM
gif_filename = "sz_gifs/sz_time_evolution_N{}_S{}.gif".format(N_sites, state_ind)
if os.path.exists(gif_filename):
    os.remove(gif_filename)

writer = imageio.get_writer(gif_filename, mode="I", duration=0.2)

for t_idx, t_val in tqdm(enumerate(times), total=T_steps, desc="Creating GIF frames"):
    # 1) extract the 2D data slice at index t_idx
    frame_data = sz_evo[:, t_idx].reshape(Ly, Lx)

    # 2) set up a 4×4″ figure at 100 dpi → main image will be exactly 1000×1000 px
    fig = plt.figure(figsize=(10, 10), dpi=100)
    # we’ll carve out the bottom 5% of the figure for the time‐bar
    main_ax = fig.add_axes([0.05, 0.10, 0.90, 0.85])  # [left, bottom, width, height] in fraction
    bar_ax  = fig.add_axes([0.05, 0.02, 0.90, 0.05])  # tiny strip along bottom (5% high)

    # 3) draw the main |C_{jj}(t)| image
    im = main_ax.imshow(
        frame_data,
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
        interpolation="none"
    )
    main_ax.set_title(f"$t = {t_val:.2f}$")
    main_ax.axis("off")

    # 4) add a colorbar next to the main image (occupies part of that 85% height)
    cbar = fig.colorbar(im, ax=main_ax, fraction=0.03, pad=0.02)
    cbar.set_label("|C_{jj}(t)| (linear)", rotation=270, labelpad=15)

    # 5) compute fraction of “time bar” (0 at t_idx=0, 1 at t_idx=T_steps−1)
    frac = t_idx / float(T_steps - 1)

    # 6) draw the time‐bar as a filled rectangle from x=0→x=frac
    bar_ax.barh(
        y=0,
        width=frac,
        height=1.0,
        left=0,
        color="k"
    )
    bar_ax.set_xlim(0, 1)
    bar_ax.set_ylim(-0.5, +0.5)
    bar_ax.set_xticks([0, 1])
    bar_ax.set_xticklabels(
        [f"0", f"{times[-1]:.1f}"],
        fontsize=15
    )
    bar_ax.set_yticks([])
    bar_ax.set_xlabel("t", fontsize=15)
    bar_ax.tick_params(axis="x", which="both", length=0)
    bar_ax.spines["top"].set_visible(False)
    bar_ax.spines["right"].set_visible(False)
    bar_ax.spines["left"].set_visible(False)
    bar_ax.spines["bottom"].set_linewidth(0.5)

    # 7) save figure→PNG into an in‐memory buffer, then read it
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)  # ensures 400×400 px
    plt.close(fig)

    buf.seek(0)
    frame = imageio.v2.imread(buf)
    writer.append_data(frame)

writer.close()
print(f"Wrote GIF to {gif_filename}")