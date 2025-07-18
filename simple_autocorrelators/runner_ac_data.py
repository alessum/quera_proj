import os
import argparse
import numpy as np
from tqdm import tqdm
from hamiltonian_wrapper import RydbergLatticeSystem

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--bitstring-file', required=True)
parser.add_argument('--time_to_run', type=int, default=1000)
parser.add_argument('--Lx', type=int,   default=3,
                    help='Number of sites along x')
parser.add_argument('--Ly', type=int,   default=3,
                    help='Number of sites along y')
args = parser.parse_args()

print(f"[DEBUG] === runner_ac_data.py START ===")
print(f"[DEBUG] Arguments: {vars(args)}")

Lx, Ly = args.Lx, args.Ly
spacing   = 5.93
positions = [(i * spacing, j * spacing)
             for i in range(Lx) for j in range(Ly)]
N_sites = len(positions)

print(f"[DEBUG] Lattice: {Lx}x{Ly} = {N_sites} sites")
print(f"[DEBUG] Positions: {positions}")

# Hamiltonian parameters
delta_G, delta_L, C6 = 125, 0.0, 5.42e6
T_total = 4.0
# Time grid
T_steps = 200
times = np.linspace(0.0, T_total, T_steps)

print(f"[DEBUG] Hamiltonian params: delta_G={delta_G}, delta_L={delta_L}, C6={C6}")
print(f"[DEBUG] Time evolution: T_total={T_total}, T_steps={T_steps}")

# Build system
print(f"[DEBUG] Building Hamiltonian...")
system = RydbergLatticeSystem(
    positions=positions,
    h=np.zeros(N_sites),
    Delta_G=delta_G,
    Delta_L=delta_L,
    C6=C6,
    Omega_t=lambda t: 15.8,
    phi_t=lambda t: 0.0,
)
H = system.build_hamiltonian()
print(f"[DEBUG] Hamiltonian built successfully")

# Read bitstrings from file
print(f"[DEBUG] Reading bitstrings from: {args.bitstring_file}")
entries = []
with open(args.bitstring_file) as f:
    for line in f:
        cid, bs = line.strip().split()
        entries.append((int(cid), bs))

print(f"[DEBUG] Read {len(entries)} bitstring entries")
print(f"[DEBUG] First few entries: {entries[:3]}")

# Process each initial state
for i, (cid, bitstring) in enumerate(entries):
    print(f"[DEBUG] Processing {i+1}/{len(entries)}: circuit_id={cid}, bitstring={bitstring}")
    
    if len(bitstring) != N_sites:
        raise ValueError(f"Expected bitstring length {N_sites}, got {len(bitstring)}")

    # Build psi0 from bitstring
    up = np.array([1, 0])
    down = np.array([0, 1])
    state_vectors = [up if ch=='1' else down for ch in bitstring]
    # This creates the full quantum state vector
    psi0 = state_vectors[0]
    for vec in state_vectors[1:]:
        psi0 = np.kron(psi0, vec)  # Creates 2^25 dimensional vector!

    print(f"[DEBUG] Initial state built, norm: {np.linalg.norm(psi0):.6f}")

    # Compute ZZ autocorrelator for the single excited site
    excited_site = bitstring.index('1')
    print(f"[DEBUG] Excited site: {excited_site}")
    
    corr = system.compute_zz_autocorrelator(psi0, times, sites=[excited_site], percentage=cid/len(entries))
    corr_real = np.real(corr)
    
    print(f"[DEBUG] Correlator computed, shape: {corr_real.shape}")

    # Save result
    output_dir = f"results/L{Lx}_Ly{Ly}_T{args.time_to_run}"
    out_dir = os.path.join(output_dir)
    os.makedirs(out_dir, exist_ok=True)
    fn = os.path.join(out_dir, f"correlator_circuit{cid}.csv")
    np.savetxt(fn, corr_real, delimiter=',')
    print(f"[DEBUG] Saved to: {fn}")

print(f"[DEBUG] === runner_ac_data.py END ===")