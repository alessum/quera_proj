import os
import argparse
import numpy as np
import sys
import psutil
from tqdm import tqdm
from hamiltonian_wrapper import RydbergLatticeSystem

# Usage:
# Matrix-based (higher accuracy, more memory): python runner_ac_data.py --bitstring-file file.txt
# Matrix-free (lower memory, faster): python runner_ac_data.py --bitstring-file file.txt --matrix-free

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
    return system_memory.available / 1024 / 1024

def check_memory_requirements_matrix_based(N_sites):
    """Check memory for matrix-based computation"""
    # State vector
    state_vector_mb = (2**N_sites * 16) / 1024 / 1024
    # Hamiltonian matrix (sparse)
    hamiltonian_mb = (2**N_sites * 2**N_sites * 16) / 1024 / 1024 * 0.1  # Assume 10% sparsity
    # Working space for time evolution
    total_needed_mb = state_vector_mb * 4 + hamiltonian_mb + 1000  # 4x for working space, +1GB overhead
    
    available_mb = print_memory_usage()
    
    debug_print(f"[MEMORY] Matrix-based memory requirements for {N_sites} qubits:")
    debug_print(f"[MEMORY] - State vector: {state_vector_mb:.1f} MB")
    debug_print(f"[MEMORY] - Hamiltonian matrix (sparse): {hamiltonian_mb:.1f} MB")
    debug_print(f"[MEMORY] - Working space: {total_needed_mb:.1f} MB")
    debug_print(f"[MEMORY] - Available: {available_mb:.1f} MB")
    
    if total_needed_mb > available_mb:
        debug_print(f"[ERROR] Insufficient memory: need {total_needed_mb:.1f}MB, have {available_mb:.1f}MB")
        return False
    
    return True

def check_memory_requirements_matrix_free(N_sites):
    """Check memory for matrix-free computation"""
    # Only need state vectors, NO Hamiltonian matrix
    state_vector_mb = (2**N_sites * 16) / 1024 / 1024
    
    # Matrix-free: only state vectors + working space for time evolution
    total_needed_mb = state_vector_mb * 4 + 500  # 4x for working space, +500MB overhead
    
    available_mb = print_memory_usage()
    
    debug_print(f"[MEMORY] Matrix-free memory requirements for {N_sites} qubits:")
    debug_print(f"[MEMORY] - State vector: {state_vector_mb:.1f} MB")
    debug_print(f"[MEMORY] - Working space: {total_needed_mb:.1f} MB")
    debug_print(f"[MEMORY] - Available: {available_mb:.1f} MB")
    debug_print(f"[MEMORY] - Hamiltonian matrix: NOT STORED (matrix-free)")
    
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
parser.add_argument('--matrix-free', action='store_true', help='Use matrix-free computation (default: False)')
args = parser.parse_args()

debug_print(f"[DEBUG] === runner_ac_data.py START ({'MATRIX-FREE' if args.matrix_free else 'MATRIX-BASED'}) ===")
debug_print(f"[DEBUG] Arguments: {vars(args)}")

Lx, Ly = args.Lx, args.Ly
N_sites = Lx * Ly

debug_print(f"[DEBUG] System size check: {N_sites} sites")

# Check memory requirements based on chosen approach
if args.matrix_free:
    if not check_memory_requirements_matrix_free(N_sites):
        debug_print(f"[ERROR] Exiting due to insufficient memory for matrix-free approach")
        sys.exit(1)
    debug_print(f"[DEBUG] Matrix-free memory check passed!")
else:
    if not check_memory_requirements_matrix_based(N_sites):
        debug_print(f"[ERROR] Exiting due to insufficient memory for matrix-based approach")
        debug_print(f"[HINT] Try using --matrix-free flag for lower memory usage")
        sys.exit(1)
    debug_print(f"[DEBUG] Matrix-based memory check passed!")

spacing = 5.93
positions = [(i * spacing, j * spacing) for i in range(Lx) for j in range(Ly)]

debug_print(f"[DEBUG] Lattice: {Lx}x{Ly} = {N_sites} sites")

# Hamiltonian parameters
delta_G, delta_L, C6 = 125, 0.0, 5.42e6
T_total = 4.0
T_steps = 200
times = np.linspace(0.0, T_total, T_steps)

debug_print(f"[DEBUG] Creating RydbergLatticeSystem...")
system = RydbergLatticeSystem(
    positions=positions,
    h=np.zeros(N_sites),
    Delta_G=delta_G,
    Delta_L=delta_L,
    C6=C6,
    Omega_t=lambda t: 15.8,
    phi_t=lambda t: 0.0,
)
debug_print(f"[DEBUG] System created successfully")

# Build Hamiltonian if not using matrix-free approach
if not args.matrix_free:
    debug_print(f"[DEBUG] Building Hamiltonian matrix...")
    system.build_hamiltonian()
    debug_print(f"[DEBUG] Hamiltonian built successfully")

print_memory_usage()

# Read bitstrings
debug_print(f"[DEBUG] Reading bitstrings...")
entries = []
with open(args.bitstring_file) as f:
    for line in f:
        cid, bs = line.strip().split()
        entries.append((int(cid), bs))

debug_print(f"[DEBUG] Read {len(entries)} bitstring entries")

# Process each state
for i, (cid, bitstring) in enumerate(entries):
    debug_print(f"[DEBUG] Processing {i+1}/{len(entries)}: circuit_id={cid}")
    
    try:
        # Build quantum state
        up = np.array([1, 0])
        down = np.array([0, 1])
        state_vectors = [up if ch=='1' else down for ch in bitstring]
        
        psi0 = state_vectors[0]
        for j, vec in enumerate(state_vectors[1:], 1):
            psi0 = np.kron(psi0, vec)
            if j in [10, 20] or j == len(state_vectors) - 1:
                debug_print(f"[DEBUG] State size after {j+1} sites: {psi0.nbytes/1024/1024:.1f} MB")
        
        debug_print(f"[DEBUG] State built: norm={np.linalg.norm(psi0):.6f}")
        print_memory_usage()
        
        # Compute autocorrelator using chosen method
        excited_site = bitstring.index('1')
        
        if args.matrix_free:
            debug_print(f"[DEBUG] Computing zero-matrix autocorrelator for site {excited_site}")
            corr = system.compute_zz_autocorrelator_zero_matrices(psi0, times, sites=[excited_site])
            debug_print(f"[DEBUG] Autocorrelator computed (matrix-free)")
        else:
            debug_print(f"[DEBUG] Computing matrix-based autocorrelator for site {excited_site}")
            # Calculate percentage based on current progress
            progress_pct = ((i + 1) / len(entries)) * 100
            corr = system.compute_zz_autocorrelator(psi0, times, sites=[excited_site], percentage=progress_pct)
            debug_print(f"[DEBUG] Autocorrelator computed (matrix-based)")
            
        print_memory_usage()
        
        # Save result
        output_dir = f"results/L{Lx}_Ly{Ly}_T{args.time_to_run}{'_matrix_free' if args.matrix_free else '_matrix_based'}"
        os.makedirs(output_dir, exist_ok=True)
        fn = os.path.join(output_dir, f"correlator_circuit{cid}.csv")
        np.savetxt(fn, np.real(corr), delimiter=',')
        debug_print(f"[DEBUG] Saved: {fn}")
        
    except MemoryError as e:
        debug_print(f"[ERROR] Out of memory for circuit {cid}: {e}")
        sys.exit(1)
    except Exception as e:
        debug_print(f"[ERROR] Failed processing circuit {cid}: {e}")
        import traceback
        debug_print(f"[ERROR] Traceback: {traceback.format_exc()}")
        continue

debug_print(f"[DEBUG] === runner_ac_data.py END ({'MATRIX-FREE' if args.matrix_free else 'MATRIX-BASED'}) ===")