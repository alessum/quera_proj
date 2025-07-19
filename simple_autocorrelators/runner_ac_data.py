import os
import argparse
import numpy as np
import sys
import psutil
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
    return system_memory.available / 1024 / 1024  # Return available MB

def check_memory_requirements(N_sites):
    """Check if we have enough memory for the computation"""
    # Each complex number = 16 bytes (8 for real + 8 for imaginary)
    state_vector_mb = (2**N_sites * 16) / 1024 / 1024
    
    # Hamiltonian matrix is 2^N x 2^N complex numbers
    hamiltonian_mb = (2**N_sites * 2**N_sites * 16) / 1024 / 1024
    
    # Total memory needed (rough estimate)
    total_needed_mb = state_vector_mb + hamiltonian_mb + 500  # +500MB for overhead
    
    available_mb = print_memory_usage()
    
    debug_print(f"[MEMORY] Memory requirements for {N_sites} qubits:")
    debug_print(f"[MEMORY] - State vector: {state_vector_mb:.1f} MB")
    debug_print(f"[MEMORY] - Hamiltonian matrix: {hamiltonian_mb:.1f} MB")
    debug_print(f"[MEMORY] - Total needed: {total_needed_mb:.1f} MB")
    debug_print(f"[MEMORY] - Available: {available_mb:.1f} MB")
    
    if total_needed_mb > available_mb:
        debug_print(f"[ERROR] Insufficient memory: need {total_needed_mb:.1f}MB, have {available_mb:.1f}MB")
        return False
    
    return True

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--bitstring-file', required=True)
parser.add_argument('--time_to_run', type=int, default=1000)
parser.add_argument('--Lx', type=int, default=3, help='Number of sites along x')
parser.add_argument('--Ly', type=int, default=3, help='Number of sites along y')
args = parser.parse_args()

debug_print(f"[DEBUG] === runner_ac_data.py START ===")
debug_print(f"[DEBUG] Arguments: {vars(args)}")

# Add early exit for large systems
Lx, Ly = args.Lx, args.Ly
N_sites = Lx * Ly

debug_print(f"[DEBUG] System size check: {N_sites} sites")

# Check memory requirements BEFORE attempting computation
if not check_memory_requirements(N_sites):
    debug_print(f"[ERROR] Exiting due to insufficient memory")
    sys.exit(1)

debug_print(f"[DEBUG] Memory check passed, proceeding with computation...")

spacing = 5.93
positions = [(i * spacing, j * spacing) for i in range(Lx) for j in range(Ly)]

debug_print(f"[DEBUG] Lattice: {Lx}x{Ly} = {N_sites} sites")
debug_print(f"[DEBUG] Positions: {positions[:3]}...") # Only show first few

# Hamiltonian parameters
delta_G, delta_L, C6 = 125, 0.0, 5.42e6
T_total = 4.0
T_steps = 200
times = np.linspace(0.0, T_total, T_steps)

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

# Read and process bitstrings...
debug_print(f"[DEBUG] Reading bitstrings from: {args.bitstring_file}")
entries = []
with open(args.bitstring_file) as f:
    for line in f:
        cid, bs = line.strip().split()
        entries.append((int(cid), bs))

debug_print(f"[DEBUG] Read {len(entries)} bitstring entries")

for i, (cid, bitstring) in enumerate(entries):
    debug_print(f"[DEBUG] Processing {i+1}/{len(entries)}: circuit_id={cid}")
    
    # Build quantum state with memory monitoring
    try:
        up = np.array([1, 0])
        down = np.array([0, 1])
        state_vectors = [up if ch=='1' else down for ch in bitstring]
        
        psi0 = state_vectors[0]
        for j, vec in enumerate(state_vectors[1:], 1):
            psi0 = np.kron(psi0, vec)
            
        debug_print(f"[DEBUG] State built: norm={np.linalg.norm(psi0):.6f}, size={psi0.nbytes/1024/1024:.1f}MB")
        
        # Continue with computation...
        excited_site = bitstring.index('1')
        corr = system.compute_zz_autocorrelator(psi0, times, sites=[excited_site])
        
        # Save result
        output_dir = f"results/L{Lx}_Ly{Ly}_T{args.time_to_run}"
        os.makedirs(output_dir, exist_ok=True)
        fn = os.path.join(output_dir, f"correlator_circuit{cid}.csv")
        np.savetxt(fn, np.real(corr), delimiter=',')
        debug_print(f"[DEBUG] Saved: {fn}")
        
    except MemoryError as e:
        debug_print(f"[ERROR] Out of memory for circuit {cid}: {e}")
        continue
    except Exception as e:
        debug_print(f"[ERROR] Failed processing circuit {cid}: {e}")
        continue

debug_print(f"[DEBUG] === runner_ac_data.py END ===")