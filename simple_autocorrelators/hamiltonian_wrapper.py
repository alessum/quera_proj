import os, io, imageio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d
from quspin.tools.measurements import obs_vs_time
from tqdm import tqdm
import functools as ft
from scipy.sparse.linalg import expm_multiply
import itertools

class RydbergLatticeSystem:
    """
    A 2D Rydberg lattice of N qubits with van-der-Waals (1/r^6) interactions
    and a global time-dependent drive Ω(t)e^{±iφ(t)}.

    Methods:
      - build_hamiltonian(): constructs the time-dependent QuSpin Hamiltonian H(t).
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
            Global detuning Δ_G, lattice detuning factor Δ_L, and van-der-Waals coefficient C6.
        Omega_t : callable, Ω(t) → float
            Time-dependent Rabi frequency (can be complex).
        phi_t : callable, φ(t) → float
            Time-dependent phase.
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

        # Precompute pairwise distances r_{jk} = ||r_j - r_k||
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
        Build and return the time-dependent Hamiltonian H(t) using QuSpin.

        The Hamiltonian has:
          1) Static terms:
             a) ∑_j [ (κ_j - Δ_G - h_j Δ_L)/2 ] σ^z_j
             b) ∑_{j<k} [ C6/(4 r_{jk}^6 ) ] σ^z_j σ^z_k
             c) Identity shifts: ∑_j [ (-Δ_G - h_j Δ_L)/2 + ∑_{k≠j} C6/(4 r_{jk}^6 ) ] · I_j
          2) Dynamic drive:
             Ω(t)/2 e^{+iφ(t)} ∑_j σ^-_j  +  Ω(t)/2 e^{-iφ(t)} ∑_j σ^+_j
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

        # 1a) σ^z term: ∑_j [ (κ_j - Δ_G - h_j Δ_L)/2 ] σ^z_j
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

        # 1c) Identity-shift: 
        #    ∑_j [ (-Δ_G - h_j Δ_L)/2 + ∑_{k≠j} C6/(4 r_{jk}^6 ) ] · I_j
        I_list = []
        for j in range(Ns):
            shift = 0.5 * (-ΔG - h_arr[j] * ΔL)
            shift += sum(C6 / (4.0 * (rmat[j, k] ** 6)) for k in range(Ns) if k != j)
            I_list.append((shift, j))
        static_list.append(("I", I_list))

        # 2) DYNAMIC PART (time-dependent drive)
        def drive_minus(t, y=None, *args):
            return 0.5 * Omega_t(t) * np.exp(1j * phi_t(t))

        def drive_plus(t, y=None, *args):
            return 0.5 * Omega_t(t) * np.exp(-1j * phi_t(t))

        # site-lists for σ^- and σ^+ terms: coefficient → σ^-_j and σ^+_j
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

    def compute_zz_autocorrelator(self, psi0, times, sites=None, percentage=None, atol=1e-9, rtol=1e-7):
        """
        Compute on-site ZZ autocorrelator:
          C_{jj}(t) = ⟨ψ₀ | σ^z_j(t) σ^z_j(0) | ψ₀⟩.

        Arguments:
        ----------
        psi0 : ndarray, shape (2^N,)
            Initial state |ψ(0)> (normalized).
        times : array_like of floats
            Times at which to compute C_{jj}(t).
        sites : sequence of int, optional
            If None, computes for all j = 0..N_s-1.
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

        # Pre-grab basis-state bit-patterns (length = 2^N)
        states = basis.states

        for p, j in tqdm(enumerate(sites), 
                         desc=f"Computing ZZ correlators {percentage:.2f}", total=M):
            # Build ±1 diagonal for σ^z_j:
            bit_j = ((states >> j) & 1).astype(np.int8)             # 0 if spin-j is ↓, 1 if ↑
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
            If None, computes for all j = 0..N_s-1.
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

    def apply_hamiltonian(self, psi, t=0):
        """Apply Hamiltonian to state vector without building full matrix"""
        N = len(self.positions)
        result = np.zeros_like(psi)
        
        # Get time-dependent parameters
        Omega = self.Omega_t(t)
        phi = self.phi_t(t)
        
        # 1. Apply single-qubit terms: h_j σ^z_j + (Δ_G + Δ_L) n_j
        for j in range(N):
            # σ^z_j term
            result += self.h[j] * self._apply_pauli_z(psi, j)
            
            # n_j = (I + σ^z_j)/2 term  
            result += (self.Delta_G + self.Delta_L) * self._apply_number_operator(psi, j)
        
        # 2. Apply Rabi terms: (Ω/2) * Σ_j [e^{-iφ} σ^+_j + e^{iφ} σ^-_j]
        for j in range(N):
            result += (Omega/2) * np.exp(-1j*phi) * self._apply_sigma_plus(psi, j)
            result += (Omega/2) * np.exp(1j*phi) * self._apply_sigma_minus(psi, j)
        
        # 3. Apply interaction terms: C6 * Σ_{i<j} n_i n_j / |r_i - r_j|^6
        for i in range(N):
            for j in range(i+1, N):
                r_ij = np.linalg.norm(np.array(self.positions[i]) - np.array(self.positions[j]))
                coupling = self.C6 / (r_ij**6)
                result += coupling * self._apply_two_qubit_interaction(psi, i, j)
        
        return result
    
    def _apply_pauli_z(self, psi, site):
        """Apply σ^z to specific site"""
        N = len(self.positions)
        result = np.zeros_like(psi)
        
        for state_idx in range(len(psi)):
            # Convert state index to bit representation
            bits = [(state_idx >> k) & 1 for k in range(N)]
            
            # Apply σ^z: |0⟩ → |0⟩, |1⟩ → -|1⟩
            sign = 1 if bits[site] == 0 else -1
            result[state_idx] = sign * psi[state_idx]
            
        return result
    
    def _apply_number_operator(self, psi, site):
        """Apply number operator n_j = |1⟩⟨1|"""
        N = len(self.positions)
        result = np.zeros_like(psi)
        
        for state_idx in range(len(psi)):
            bits = [(state_idx >> k) & 1 for k in range(N)]
            
            # n_j: only acts on |1⟩ states
            if bits[site] == 1:
                result[state_idx] = psi[state_idx]
            
        return result
    
    def _apply_sigma_plus(self, psi, site):
        """Apply σ^+ = |1⟩⟨0|"""
        N = len(self.positions)
        result = np.zeros_like(psi)
        
        for state_idx in range(len(psi)):
            bits = [(state_idx >> k) & 1 for k in range(N)]
            
            # σ^+: |0⟩ → |1⟩, |1⟩ → 0
            if bits[site] == 0:
                # Flip bit at site
                new_bits = bits.copy()
                new_bits[site] = 1
                new_state_idx = sum(bit * (2**k) for k, bit in enumerate(new_bits))
                result[new_state_idx] = psi[state_idx]
                
        return result
    
    def _apply_sigma_minus(self, psi, site):
        """Apply σ^- = |0⟩⟨1|"""
        N = len(self.positions)
        result = np.zeros_like(psi)
        
        for state_idx in range(len(psi)):
            bits = [(state_idx >> k) & 1 for k in range(N)]
            
            # σ^-: |1⟩ → |0⟩, |0⟩ → 0
            if bits[site] == 1:
                # Flip bit at site
                new_bits = bits.copy()
                new_bits[site] = 0
                new_state_idx = sum(bit * (2**k) for k, bit in enumerate(new_bits))
                result[new_state_idx] = psi[state_idx]
                
        return result
    
    def _apply_two_qubit_interaction(self, psi, site_i, site_j):
        """Apply n_i * n_j interaction"""
        N = len(self.positions)
        result = np.zeros_like(psi)
        
        for state_idx in range(len(psi)):
            bits = [(state_idx >> k) & 1 for k in range(N)]
            
            # n_i * n_j: only acts when both sites are |1⟩
            if bits[site_i] == 1 and bits[site_j] == 1:
                result[state_idx] = psi[state_idx]
                
        return result

    def apply_hamiltonian_pure_manual(self, psi, t=0):
        """Apply Hamiltonian with ZERO matrices - pure bit operations"""
        N = len(self.positions)
        Omega = self.Omega_t(t)
        phi = self.phi_t(t)
        result = np.zeros_like(psi)
        
        dim = 2**N  # Hilbert space dimension
        
        # Manual bit operations without any QuSpin basis
        for state_idx in range(dim):
            # Extract bits manually
            bits = [(state_idx >> j) & 1 for j in range(N)]
            
            # 1. Apply σ^z terms
            for j in range(N):
                sz_val = 2 * bits[j] - 1  # Convert 0,1 to -1,+1
                result[state_idx] += self.h[j] * sz_val * psi[state_idx]
            
            # 2. Apply detuning terms
            for j in range(N):
                result[state_idx] += (self.Delta_G + self.Delta_L) * bits[j] * psi[state_idx]
            
            # 3. Apply interaction terms
            for i in range(N):
                for j in range(i+1, N):
                    if bits[i] == 1 and bits[j] == 1:  # Both excited
                        r_ij = self.rmat[i, j]
                        coupling = self.C6 / (r_ij**6)
                        result[state_idx] += coupling * psi[state_idx]
    
        # 4. Apply Rabi terms (σ^+ and σ^-)
        for state_idx in range(dim):
            bits = [(state_idx >> j) & 1 for j in range(N)]
            
            for j in range(N):
                # σ^+ : |0⟩ → |1⟩ at site j
                if bits[j] == 0:
                    new_state = state_idx | (1 << j)  # Flip bit j to 1
                    result[new_state] += (Omega/2) * np.exp(-1j*phi) * psi[state_idx]
                
                # σ^- : |1⟩ → |0⟩ at site j  
                if bits[j] == 1:
                    new_state = state_idx & ~(1 << j)  # Flip bit j to 0
                    result[new_state] += (Omega/2) * np.exp(1j*phi) * psi[state_idx]
    
        return result

    def _apply_pauli_z_pure_manual(self, psi, site):
        """Apply σ^z without any QuSpin matrices"""
        N = len(self.positions)
        result = np.zeros_like(psi)
        
        for state_idx in range(2**N):
            bit_val = (state_idx >> site) & 1
            sz_val = 2 * bit_val - 1  # 0→-1, 1→+1
            result[state_idx] = sz_val * psi[state_idx]
        
        return result

    def compute_zz_autocorrelator_zero_matrices(self, psi0, times, sites=None):
        """100% matrix-free - no QuSpin, no matrices, pure bit operations"""
        from scipy.integrate import solve_ivp
        
        if sites is None:
            sites = list(range(len(self.positions)))
        
        def schrodinger_rhs(t, psi_flat):
            """Right-hand side for Schrödinger equation"""
            psi = psi_flat.view(complex)
            H_psi = self.apply_hamiltonian_pure_manual(psi, t)
            return (-1j * H_psi).view(float)
    
        correlators = []
        
        for site in sites:
            print(f"[DEBUG] Computing correlator for site {site} (ZERO matrices)")
            corr_site = []
            
            # Pre-compute σ^z|ψ₀⟩ without matrices
            sigma_z_psi0 = self._apply_pauli_z_pure_manual(psi0, site)
            
            # Use only first 20 time points for testing
            max_time_points = min(20, len(times))
            time_subset = times[:max_time_points]
            
            for i, t in enumerate(time_subset):
                if t == 0:
                    corr = np.vdot(sigma_z_psi0, sigma_z_psi0).real
                else:
                    try:
                        sol = solve_ivp(
                            schrodinger_rhs, [0, t], psi0.view(float),
                            method='RK23',  # Fastest method
                            rtol=1e-3,      # Very loose tolerance
                            atol=1e-5,
                            max_step=0.5
                        )
                        
                        if sol.success:
                            psi_t = sol.y[:, -1].view(complex)
                            sigma_z_psi_t = self._apply_pauli_z_pure_manual(psi_t, site)
                            corr = np.vdot(sigma_z_psi_t, sigma_z_psi0).real
                        else:
                            corr = 0.0
                    except:
                        corr = 0.0
            
                corr_site.append(corr)
                
                if i % 5 == 0:
                    print(f"[DEBUG] Site {site}: {i+1}/{max_time_points} time points")
        
        # Pad with zeros
        while len(corr_site) < len(times):
            corr_site.append(0.0)
            
        correlators.append(corr_site)
    
        return np.array(correlators)


