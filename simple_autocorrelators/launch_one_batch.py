#!/usr/bin/env python3
import argparse, itertools, os, subprocess, sys, tempfile

def debug_print(msg):
    """Print with immediate flush for real-time output"""
    print(msg)
    sys.stdout.flush()

def generate_valid_bitstrings(N, mid_site, start_idx, count):
    """Generate valid bitstrings on-demand without storing all in memory"""
    debug_print(f"[DEBUG] Starting bitstring generation: N={N}, mid_site={mid_site}, start={start_idx}, count={count}")
    
    current_idx = 0
    target_end = start_idx + count
    generated = 0
    
    for i, bits in enumerate(itertools.product('01', repeat=N)):
        # Progress indicator for large iterations
        if i % 1000000 == 0 and i > 0:
            debug_print(f"[DEBUG] Processed {i:,} combinations, found {current_idx} valid, generated {generated}")
        
        if bits[mid_site] == '1':  # Valid bitstring
            if current_idx >= start_idx and current_idx < target_end:
                yield current_idx, ''.join(bits)
                generated += 1
            current_idx += 1
            if current_idx >= target_end:
                debug_print(f"[DEBUG] Reached target, stopping generation")
                break
    
    debug_print(f"[DEBUG] Generation complete: {generated} bitstrings generated")

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

    debug_print(f"[DEBUG] === launch_one_batch.py START ===")
    debug_print(f"[DEBUG] Arguments: {vars(args)}")

    # total sites and all valid bitstrings:
    N = args.Lx * args.Ly
    debug_print(f"[DEBUG] Lx={args.Lx}, Ly={args.Ly} → N_sites={N}")
    
    # Add memory warning for large lattices
    if N > 20:
        debug_print(f"[WARNING] Large lattice detected (N={N}). This may require significant memory.")
        debug_print(f"[WARNING] Total Hilbert space size: 2^{N} = {2**N:,}")
    
    debug_print(f"[DEBUG] mid_site={args.mid_site} (must be < {N})")
    
    # Check if mid_site is valid
    if args.mid_site >= N:
        raise ValueError(f"mid_site ({args.mid_site}) must be < N_sites ({N})")
    
    debug_print(f"[DEBUG] About to calculate total_valid...")
    # Don't create all bitstrings - just count them
    total_valid = 2**(N-1)  # Mathematical formula instead of generation
    debug_print(f"[DEBUG] Total valid bitstrings: {total_valid:,}")

    start = args.batch_index * args.batch_size
    count = min(args.batch_size, total_valid - start)
    
    debug_print(f"[DEBUG] Batch {args.batch_index}: indices {start}..{start+count-1} ({count} states)")

    # Generate only the bitstrings we need
    debug_print(f"[DEBUG] Creating temporary file...")
    with tempfile.NamedTemporaryFile('w', delete=False) as tmp:
        tmp_path = tmp.name
        debug_print(f"[DEBUG] Temp file created: {tmp_path}")
        
        debug_print(f"[DEBUG] Starting bitstring generation loop...")
        count_written = 0
        for idx, bitstring in generate_valid_bitstrings(N, args.mid_site, start, count):
            tmp.write(f"{idx} {bitstring}\n")
            count_written += 1
            if count_written == 1:
                debug_print(f"[DEBUG] First bitstring written: {idx} {bitstring}")
            elif count_written % 5 == 0:
                debug_print(f"[DEBUG] Written {count_written} bitstrings so far...")
        
        debug_print(f"[DEBUG] Finished writing {count_written} bitstrings to temp file")
    
    debug_print(f"[DEBUG] About to call runner script...")
    debug_print(f"[DEBUG] Runner script path: {args.runner_script}")
    
    try:
        debug_print(f"[DEBUG] Starting subprocess...")
        result = subprocess.run([
            sys.executable, args.runner_script,
            '--bitstring-file', tmp_path,
            '--time_to_run',     str(args.time),
            '--Lx',              str(args.Lx),
            '--Ly',              str(args.Ly),
        ], check=True, capture_output=True, text=True, timeout=300)  # Reduce to 5 min timeout
        
        debug_print(f"[DEBUG] Runner completed successfully")
        if result.stdout:
            debug_print(f"[DEBUG] Runner stdout: {result.stdout}")
            
    except subprocess.TimeoutExpired:
        debug_print(f"[ERROR] Runner timed out after 5 minutes")
        raise
    except subprocess.CalledProcessError as e:
        debug_print(f"[ERROR] Runner failed with return code {e.returncode}")
        debug_print(f"[ERROR] stderr: {e.stderr}")
        raise
    finally:
        debug_print(f"[DEBUG] Cleaning up temp file...")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            debug_print(f"[DEBUG] Temp file removed")
    
    debug_print(f"[DEBUG] === launch_one_batch.py END ===")

if __name__=='__main__':
    main()
