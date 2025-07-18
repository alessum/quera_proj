#!/usr/bin/env python3
import argparse, itertools, os, subprocess, sys, tempfile

def generate_valid_bitstrings(N, mid_site, start_idx, count):
    """Generate valid bitstrings on-demand without storing all in memory"""
    current_idx = 0
    target_end = start_idx + count
    
    for bits in itertools.product('01', repeat=N):
        if bits[mid_site] == '1':  # Valid bitstring
            if current_idx >= start_idx and current_idx < target_end:
                yield current_idx, ''.join(bits)
            current_idx += 1
            if current_idx >= target_end:
                break

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--time',         type=int, required=True)
    p.add_argument('--mid_site',     type=int, required=True)
    p.add_argument('--Lx',           type=int, required=True)
    p.add_argument('--Ly',           type=int, required=True)
    p.add_argument('--batch_size',   type=int, required=True)
    p.add_argument('--batch_index',  type=int, required=True)
    p.add_argument('--runner-script',type=str,
                   default="simple_autocorrelators/runner_ac_data.py")
    args = p.parse_args()

    print(f"[DEBUG] === launch_one_batch.py START ===")
    print(f"[DEBUG] Arguments: {vars(args)}")

    # total sites and all valid bitstrings:
    N = args.Lx * args.Ly
    print(f"[DEBUG] Lx={args.Lx}, Ly={args.Ly} → N_sites={N}")
    
    # Add memory warning for large lattices
    if N > 20:
        print(f"[WARNING] Large lattice detected (N={N}). This may require significant memory.")
        print(f"[WARNING] Total Hilbert space size: 2^{N} = {2**N:,}")
    
    print(f"[DEBUG] mid_site={args.mid_site} (must be < {N})")
    
    # Check if mid_site is valid
    if args.mid_site >= N:
        raise ValueError(f"mid_site ({args.mid_site}) must be < N_sites ({N})")
    
    print(f"[DEBUG] Generating valid bitstrings...")
    # Don't create all bitstrings - just count them
    total_valid = 2**(N-1)  # Mathematical formula instead of generation
    print(f"[DEBUG] Total valid bitstrings: {total_valid}")

    start = args.batch_index * args.batch_size
    count = min(args.batch_size, total_valid - start)
    
    print(f"[DEBUG] Batch {args.batch_index}: indices {start}..{start+count-1} ({count} states)")

    # Generate only the bitstrings we need
    with tempfile.NamedTemporaryFile('w', delete=False) as tmp:
        for idx, bitstring in generate_valid_bitstrings(N, args.mid_site, start, count):
            tmp.write(f"{idx} {bitstring}\n")
        tmp_path = tmp.name
    
    print(f"[DEBUG] Generated bitstring length = {len(bitstring)}")
    print(f"[DEBUG] Temp file: {tmp_path}")
    assert args.mid_site < N, f"[DEBUG] mid_site ({args.mid_site}) must be less than N_sites ({N})"

    print(f"[DEBUG] Calling runner with: {args.runner_script}")
    try:
        result = subprocess.run([
            sys.executable, args.runner_script,
            '--bitstring-file', tmp_path,
            '--time_to_run',     str(args.time),
            '--Lx',              str(args.Lx),
            '--Ly',              str(args.Ly),
        ], check=True, capture_output=True, text=True, timeout=9000)  # 150 min timeout
        
        print(f"[DEBUG] Runner completed successfully")
        if result.stdout:
            print(f"[DEBUG] Runner stdout: {result.stdout}")
            
    except subprocess.TimeoutExpired:
        print(f"[ERROR] Runner timed out after 150 minutes")
        raise
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Runner failed with return code {e.returncode}")
        print(f"[ERROR] stderr: {e.stderr}")
        raise
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    print(f"[DEBUG] === launch_one_batch.py END ===")


if __name__=='__main__':
    main()
