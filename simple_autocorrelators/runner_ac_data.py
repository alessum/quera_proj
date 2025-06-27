import os
import argparse
import numpy as np
from tqdm import tqdm
from hamiltonian_wrapper import RydbergLatticeSystem

# Define lattice
Lx, Ly = 3, 3
spacing = 5.93
positions = [(i * spacing, j * spacing) for i in range(Lx) for j in range(Ly)]
N_sites = len(positions)

# Hamiltonian parameters
Delta_G, Delta_L, C6 = 125, 0.0, 5.42e6
T_total = 4.0

# Time-dependent drives
def Omega_t(t): return 15.8 + 0.0 * t
def phi_t(t): return 0.0 + 0.0 * t

# Build system
system = RydbergLatticeSystem(
    positions=positions,
    h=np.zeros(N_sites),
    Delta_G=Delta_G,
    Delta_L=Delta_L,
    C6=C6,
    Omega_t=Omega_t,
    phi_t=phi_t,
)
H = system.build_hamiltonian()

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--bitstring-file', help='File with lines: <circuit-id> <bitstring>')
parser.add_argument('--theta_to_run', type=float, default=0.0)
parser.add_argument('--time_to_run', type=int, default=1000)
parser.add_argument('--output-dir', required=True)
args = parser.parse_args()

# Read bitstrings
entries = []
with open(args.bitstring_file) as f:
    for line in f:
        cid, bs = line.strip().split()
        entries.append((int(cid), bs))

# Time grid
T_steps = 200
times = np.linspace(0.0, T_total, T_steps)

# Process each entry
for circuit_id, bitstring in entries:
    if len(bitstring) != N_sites:
        raise ValueError(f"Expected bitstring length {N_sites}, got {len(bitstring)}")

    # Build psi0
    up = np.array([1, 0])
    down = np.array([0, 1])
    state_vectors = [up if ch=='1' else down for ch in bitstring]
    psi0 = state_vectors[0]
    for vec in state_vectors[1:]:
        psi0 = np.kron(psi0, vec)

    # Compute correlator\    
    corr = system.compute_zz_autocorrelator(psi0, times, sites=[bitstring.index('1')])
    corr_real = np.real(corr)

    # Save CSV
    out_dir = os.path.join(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    fn = os.path.join(out_dir, f"correlator_L{Lx}_Ly{Ly}_circuit{circuit_id}.csv")
    np.savetxt(fn, corr_real, delimiter=',')