import os, io, imageio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d
from quspin.tools.measurements import obs_vs_time
from tqdm import tqdm
import functools as ft

class RydbergLatticeSystem:
    """
    A 2D Rydberg lattice of N qubits with van‐der‐Waals (1/r^6) interactions
    and a global time‐dependent drive Ω(t)e^{±iφ(t)}.

    Methods:
      - build_hamiltonian(): constructs the time‐dependent QuSpin Hamiltonian H(t).
      - plot_lattice(): visualizes the (x,y) coordinates of each qubit.
      - compute_zz_autocorrelator(): computes C_{jj}(t) = ⟨σ^z_j(t) σ^z_j(0)⟩.
    """

    def __init__(self, positions, h, Delta_G, Delta_L, C6, Omega_t, phi_t):
        """
        Initialize the Rydberg lattice.

        Arguments:
        ----------
        positions : sequence of length N
            Each element is a tuple (x_j, y_j) giving the 2D coordinates of qubit j.
        kappa : array_like of length N
            κ_j detuning for each site.
        h : array_like of length N
            Helmholtz factor h_j for each site.
        Delta_G, Delta_L, C6 : float
            Global detuning Δ_G, lattice detuning factor Δ_L, and van‐der‐Waals coefficient C6.
        Omega_t : callable, Ω(t) → float
            Time‐dependent Rabi frequency (can be complex).
        phi_t : callable, φ(t) → float
            Time‐dependent phase.
        """
        self.positions = np.array(positions, dtype=float)  # shape = (N, 2)
        self.Ns = len(positions)
        self.h = np.array(h, dtype=float)
        self.Delta_G = float(Delta_G)
        self.Delta_L = float(Delta_L)
        self.C6 = float(C6)
        self.Omega_t = Omega_t
        self.phi_t = phi_t

        # Construct a spin_basis_1d with pauli=True so that σ^z, σ^+, σ^- are available
        self.basis = spin_basis_1d(L=self.Ns, pauli=True)

        # Precompute pairwise distances r_{jk} = ||r_j − r_k||
        coords = self.positions
        self.rmat = np.full((self.Ns, self.Ns), np.inf, dtype=float)
        for i in range(self.Ns):
            for j in range(i + 1, self.Ns):
                dij = np.linalg.norm(coords[i] - coords[j])
                self.rmat[i, j] = dij
                self.rmat[j, i] = dij
        
        self.H = None

    def build_hamiltonian(self):
        """
        Build and return the time‐dependent Hamiltonian H(t) using QuSpin.

        The Hamiltonian has:
          1) Static terms:
             a) ∑_j [ (κ_j − Δ_G − h_j Δ_L)/2 ] σ^z_j
             b) ∑_{j<k} [ C6/(4 r_{jk}^6 ) ] σ^z_j σ^z_k
             c) Identity shifts: ∑_j [ (−Δ_G − h_j Δ_L)/2 + ∑_{k≠j} C6/(4 r_{jk}^6 ) ] · I_j
          2) Dynamic drive:
             Ω(t)/2 e^{+iφ(t)} ∑_j σ^-_j  +  Ω(t)/2 e^{−iφ(t)} ∑_j σ^+_j
        """
        Ns, h_arr = self.Ns, self.h
        ΔG, ΔL, C6 = self.Delta_G, self.Delta_L, self.C6
        Omega_t, phi_t = self.Omega_t, self.phi_t
        basis, rmat = self.basis, self.rmat

        # Precomputing the kappa terms
        kappa = np.zeros(Ns, dtype=float)
        for j in range(Ns):
            for k in range(Ns):
                if j != k:
                    kappa[j] += C6 / (2.0 * (rmat[j, k] ** 6))

        # 1) STATIC TERMS
        static_list = []

        # 1a) σ^z term: ∑_j [ (κ_j − Δ_G − h_j Δ_L)/2 ] σ^z_j
        sz_list = [
            (0.5 * (kappa[j] - ΔG - h_arr[j] * ΔL), j)
            for j in range(Ns)
        ]
        static_list.append(("z", sz_list))

        # 1b) σ^z σ^z interactions: ∑_{j<k} [ C6/(4 r_{jk}^6) ] σ^z_j σ^z_k
        zz_list = []
        for j in range(Ns):
            for k in range(j + 1, Ns):
                pref = C6 / (4.0 * (rmat[j, k] ** 6))
                zz_list.append((pref, j, k))
        static_list.append(("zz", zz_list))

        # 1c) Identity‐shift: 
        #    ∑_j [ (−Δ_G − h_j Δ_L)/2 + ∑_{k≠j} C6/(4 r_{jk}^6 ) ] · I_j
        I_list = []
        for j in range(Ns):
            shift = 0.5 * (-ΔG - h_arr[j] * ΔL)
            shift += sum(C6 / (4.0 * (rmat[j, k] ** 6)) for k in range(Ns) if k != j)
            I_list.append((shift, j))
        static_list.append(("I", I_list))

        # 2) DYNAMIC PART (time‐dependent drive)
        def drive_minus(t, y=None, *args):
            return 0.5 * Omega_t(t) * np.exp(1j * phi_t(t))

        def drive_plus(t, y=None, *args):
            return 0.5 * Omega_t(t) * np.exp(-1j * phi_t(t))

        # site‐lists for σ^- and σ^+ terms: coefficient → σ^−_j and σ^+_j
        drive_m_list = [[1, j] for j in range(Ns)]
        drive_p_list = [[1, j] for j in range(Ns)]
        dynamic_list = [
            ["-", drive_m_list, drive_minus, []],
            ["+", drive_p_list, drive_plus,  []],
        ]

        # 3) BUILD HAMILTONIAN
        self.H = hamiltonian(
            static_list,
            dynamic_list,
            basis=basis,
            dtype=np.complex128,
            check_herm=True,
            check_pcon=True,
            check_symm=True
        )
        print("Hamiltonian built successfully.")
        return self.H

    def plot_lattice(self, annotate=True, figsize=(6, 6), marker='o', color='C0'):
        """
        Plot the 2D qubit positions with optional site indices.

        Returns:
        --------
        fig, ax : matplotlib Figure and Axes
        """
        xs, ys = self.positions[:, 0], self.positions[:, 1]
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(xs, ys, s=100, marker=marker, color=color, edgecolor='k', zorder=2)
        ax.set_aspect('equal', 'box')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("2D Qubit Lattice")

        if annotate:
            for j, (xj, yj) in enumerate(self.positions):
                ax.text(xj + 0.05, yj + 0.05, str(j), fontsize=12, zorder=3)

        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        return fig, ax

    def compute_zz_autocorrelator(self, psi0, times, sites=None, atol=1e-9, rtol=1e-7):
        """
        Compute on‐site ZZ autocorrelator:
          C_{jj}(t) = ⟨ψ₀ | σ^z_j(t) σ^z_j(0) | ψ₀⟩.

        Arguments:
        ----------
        psi0 : ndarray, shape (2^N,)
            Initial state |ψ(0)> (normalized).
        times : array_like of floats
            Times at which to compute C_{jj}(t).
        sites : sequence of int, optional
            If None, computes for all j = 0..N_s−1.
        atol, rtol : float
            Tolerances for the ODE solver.

        Returns:
        --------
        corr : ndarray, shape (len(sites), len(times))
            corr[p, n] = C_{j_p j_p}(times[n]) for each site j_p.
        """
        if self.H is None:
            raise RuntimeError("Call build_hamiltonian() before computing correlators.")

        basis = self.H.basis
        N_s = basis.L
        if sites is None:
            sites = list(range(N_s))
        else:
            sites = list(sites)

        T = len(times)
        M = len(sites)
        corr = np.zeros((M, T), dtype=np.complex128)

        # Evolve |ψ₀⟩ under H(t): shape = (2^N, T)
        psi_t = self.H.evolve(psi0, 0.0, times, atol=atol, rtol=rtol)

        # Pre‐grab basis‐state bit‐patterns (length = 2^N)
        states = basis.states

        for p, j in tqdm(enumerate(sites), desc="Computing ZZ correlators", total=M):
            # Build ±1 diagonal for σ^z_j:
            bit_j = ((states >> j) & 1).astype(np.int8)             # 0 if spin‐j is ↓, 1 if ↑
            sz_vals = 2*bit_j - 1                 # map {0→-1, 1→+1}

            # |χ₀⟩ = σ^z_j |ψ₀⟩ is just elementwise multiply by ±1
            chi0 = sz_vals * psi0

            # Evolve |χ₀⟩ under H(t): shape = (2^N, T)
            chi_t = self.H.evolve(chi0, 0.0, times, atol=atol, rtol=rtol)

            for n in range(T):
                psi_n = psi_t[:, n]
                chi_n = chi_t[:, n]
                sz_chi = sz_vals * chi_n  # σ^z_j |χ(t_n)⟩
                corr[p, n] = np.vdot(psi_n, sz_chi)

        return corr
    
    def compute_sz_ev(self, psi0, times, sites=None, atol=1e-9, rtol=1e-7):
        """
        Compute the expectation value of σ^z_j(t) for each site j at times t:
        <σ^z_j(t)> = ⟨ψ₀ | σ^z_j(t) | ψ₀⟩.

        Arguments:
        ----------
        psi0 : ndarray, shape (2^N,)
            Initial state |ψ(0)> (normalized).
        times : array_like of floats
            Times at which to compute C_{jj}(t).
        sites : sequence of int, optional
            If None, computes for all j = 0..N_s−1.
        atol, rtol : float
            Tolerances for the ODE solver.

        Returns:
        --------
        sz_ev : ndarray, shape (len(sites), len(times))
        sz_ev[p, n] = <σ^z_{j_p}(t_n)> for each site j_p.
        """
        if self.H is None:
            raise RuntimeError("Call build_hamiltonian() before computing correlators.")

        basis = self.H.basis
        N_s = basis.L
        if sites is None:
            sites = list(range(N_s))
        else:
            sites = list(sites)

        T = len(times)
        M = len(sites)
        sz_ev = np.zeros((M, T), dtype=np.complex128)

        # Evolve |ψ₀⟩ under H(t): shape = (2^N, T)
        psi_t = self.H.evolve(psi0, 0.0, times, atol=atol, rtol=rtol)

        # Creating all possible local Sz operators:
        Sz_dict = {}
        """
        for j in range(N_s):
            # Build ±1 diagonal for σ^z_j:
            sz_list = [0]* self.Ns
            sz_list[j] = 1
            new_H, new_key = hamiltonian(sz_list, [], basis=basis, dtype=np.complex128), "sz_{}".format(j)
            Sz_dict[new_key] = new_H
        """
        for j in range(N_s):
            sz_term = [ [1.0, j] ]  # coefficient 1.0 at site j
            new_H = hamiltonian([("z", sz_term)], [], basis=basis, dtype=np.complex128)
            Sz_dict[f"sz_{j}"] = new_H

        sz_v_time = obs_vs_time(psi_t, times, Sz_dict, enforce_pure= True)
        sz_ev = np.zeros((M, T), dtype=np.complex128)
        for j in range(M):
            sz_ev[j, :] = sz_v_time["sz_{}".format(sites[j])]

        return sz_ev

def find_random_basis_state_j(N, j):
    """
    Find a random computational basis state index with spin j flipped to ↑.

    Arguments:
    ----------
    N : int
        Total number of spins.
    j : int
        Index of the spin to flip to ↑.

    Returns:
    --------
    state_index : ndarray, shape (2**N,)
        The computational basis state with spin j set to ↑.
    """
    up, down = np.array([1, 0]), np.array([0, 1])
    list_states = []                # Creating a list where states are randomly set to up or down
    for k in range(N):
        if k == j:
            list_states.append(up)
        else:
            randint = np.random.randint(0, 2)
            if randint == 0:
                list_states.append(down)
            else:
                list_states.append(up)
    full_state = ft.reduce(np.kron, list_states)        # Full state
    nonzero_indices = np.nonzero(full_state)[0]         
    if len(nonzero_indices) == 0:
        raise ValueError("No non-zero index found. Check the input parameters.")
    return nonzero_indices[0]
    

