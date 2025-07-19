import os
import argparse
import numpy as np
import sys
import psutil  # Add this import
from tqdm import tqdm
from hamiltonian_wrapper import RydbergLatticeSystem

def debug_print(msg):
    """Print with immediate flush for real-time output"""
    print(msg)
    sys.stdout.flush()

def print_memory_usage():
    """Print current memory usage"""
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    debug_print(f"[MEMORY] Current process memory: {memory_mb:.1f} MB")
    
    # System memory
    system_memory = psutil.virtual_memory()
    debug_print(f"[MEMORY] System: {system_memory.used/1024/1024/1024:.1f}GB used / {system_memory.total/1024/1024/1024:.1f}GB total ({system_memory.percent:.1f}%)")

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--bitstring-file', required=True)
parser.add_argument('--time_to_run', type=int, default=1000)
parser.add_argument('--Lx', type=int, default=3, help='Number of sites along x')
parser.add_argument('--Ly', type=int, default=3, help='Number of sites along y')
args = parser.parse_args()

debug_print(f"[DEBUG] === runner_ac_data.py START ===")
debug_print(f"[DEBUG] Arguments: {vars(args)}")
print_memory_usage()

# Add early exit for large systems
Lx, Ly = args.Lx, args.Ly
N_sites = Lx * Ly

debug_print(f"[DEBUG] System size check: {N_sites} sites")
debug_print(f"[DEBUG] Quantum state dimension: 2^{N_sites} = {2**N_sites:,}")
debug_print(f"[DEBUG] Memory per state: ~{(2**N_sites * 16) / 1024 / 1024:.1f} MB")

if N_sites > 25:
    debug_print(f"[ERROR] System too large: {N_sites} sites")
    sys.exit(1)

debug_print(f"[DEBUG] System size check passed: {N_sites} sites")

spacing = 5.93
positions = [(i * spacing, j * spacing) for i in range(Lx) for j in range(Ly)]
N_sites = len(positions)

debug_print(f"[DEBUG] Lattice: {Lx}x{Ly} = {N_sites} sites")
debug_print(f"[DEBUG] Positions: {positions[:3]}...") # Only show first few

# Hamiltonian parameters
delta_G, delta_L, C6 = 125, 0.0, 5.42e6
T_total = 4.0
T_steps = 200
times = np.linspace(0.0, T_total, T_steps)

debug_print(f"[DEBUG] Hamiltonian params: delta_G={delta_G}, delta_L={delta_L}, C6={C6}")
debug_print(f"[DEBUG] Time evolution: T_total={T_total}, T_steps={T_steps}")
print_memory_usage()

# Build system
debug_print(f"[DEBUG] Building Hamiltonian...")
try:
    system = RydbergLatticeSystem(
        positions=positions,
        h=np.zeros(N_sites),
        Delta_G=delta_G,
        Delta_L=delta_L,
        C6=C6,
        Omega_t=lambda t: 15.8,
        phi_t=lambda t: 0.0,
    )
    debug_print(f"[DEBUG] RydbergLatticeSystem created")
    print_memory_usage()
    
    H = system.build_hamiltonian()
    debug_print(f"[DEBUG] Hamiltonian built successfully")
    print_memory_usage()
except Exception as e:
    debug_print(f"[ERROR] Failed to build Hamiltonian: {e}")
    sys.exit(1)

# Read bitstrings from file
debug_print(f"[DEBUG] Reading bitstrings from: {args.bitstring_file}")
entries = []
with open(args.bitstring_file) as f:
    for line in f:
        cid, bs = line.strip().split()
        entries.append((int(cid), bs))

debug_print(f"[DEBUG] Read {len(entries)} bitstring entries")

# Process each initial state
for i, (cid, bitstring) in enumerate(entries):
    debug_print(f"[DEBUG] Processing {i+1}/{len(entries)}: circuit_id={cid}")
    print_memory_usage()
    
    if len(bitstring) != N_sites:
        raise ValueError(f"Expected bitstring length {N_sites}, got {len(bitstring)}")

    # Build psi0 from bitstring - THIS IS WHERE IT LIKELY FAILS
    debug_print(f"[DEBUG] Building quantum state vector...")
    try:
        up = np.array([1, 0])
        down = np.array([0, 1])
        state_vectors = [up if ch=='1' else down for ch in bitstring]
        
        psi0 = state_vectors[0]
        debug_print(f"[DEBUG] Initial state size: {psi0.shape}")
        
        for j, vec in enumerate(state_vectors[1:], 1):
            psi0 = np.kron(psi0, vec)
            if j % 5 == 0:  # Print progress every 5 sites
                debug_print(f"[DEBUG] State size after {j+1} sites: {psi0.shape}, memory: {psi0.nbytes/1024/1024:.1f} MB")
                print_memory_usage()
        
        debug_print(f"[DEBUG] Final state built, norm: {np.linalg.norm(psi0):.6f}")
        print_memory_usage()
        
    except MemoryError as e:
        debug_print(f"[ERROR] Out of memory while building quantum state: {e}")
        sys.exit(1)
    except Exception as e:
        debug_print(f"[ERROR] Failed to build quantum state: {e}")
        sys.exit(1)

    # Compute ZZ autocorrelator for the single excited site
    excited_site = bitstring.index('1')
    debug_print(f"[DEBUG] Excited site: {excited_site}")
    
    corr = system.compute_zz_autocorrelator(psi0, times, sites=[excited_site], percentage=cid/len(entries))
    corr_real = np.real(corr)
    
    debug_print(f"[DEBUG] Correlator computed, shape: {corr_real.shape}")

    # Save result
    output_dir = f"results/L{Lx}_Ly{Ly}_T{args.time_to_run}"
    out_dir = os.path.join(output_dir)
    os.makedirs(out_dir, exist_ok=True)
    fn = os.path.join(out_dir, f"correlator_circuit{cid}.csv")
    np.savetxt(fn, corr_real, delimiter=',')
    debug_print(f"[DEBUG] Saved to: {fn}")

debug_print(f"[DEBUG] === runner_ac_data.py END ===")