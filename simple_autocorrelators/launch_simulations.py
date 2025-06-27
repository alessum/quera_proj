#!/usr/bin/env python3
"""
python simple_autocorrelators/launch_simulations.py \ 
  --time 100 \
  --Lx 3 --Ly 3 \
  --batch_size 50 \
  --runner-script /Users/a.summer/Development_local/phd/quera_proj/simple_autocorrelators/runner_ac_data.py
"""

import argparse
import itertools
import os
import subprocess
import sys
import tempfile

def main():
    parser = argparse.ArgumentParser(
        description="Launch Rydberg simulations in batched bitstring groups"
    )
    parser.add_argument('--time',         type=int,   required=True,
                        help="Time to run")
    parser.add_argument('--mid_site',     type=int,   default=None,
                        help="Index of the forced-up bit (default=Ns//2)")
    parser.add_argument('--Lx',           type=int,   default=3,
                        help="Lattice X dimension")
    parser.add_argument('--Ly',           type=int,   default=3,
                        help="Lattice Y dimension")
    parser.add_argument('--batch_size',   type=int,   default=50,
                        help="How many states per batch")
    parser.add_argument('--runner-script',type=str,   default="runner_ac_data.py",
                        help="Path to the Python runner")
    args = parser.parse_args()

    # Compute sizes
    N_sites = args.Lx * args.Ly
    mid_site = args.mid_site if args.mid_site is not None else N_sites // 2

    # Enumerate all bitstrings with bit[mid_site] == '1'
    valid = [
        ''.join(bits)
        for bits in itertools.product('01', repeat=N_sites)
        if bits[mid_site] == '1'
    ]
    total = len(valid)
    print(f"Total valid bitstrings: {total}  (mid_site={mid_site})")
    print(f"Batch size: {args.batch_size} → {((total - 1)//args.batch_size + 1)} batches")

    # Launch each batch
    for batch_start in range(0, total, args.batch_size):
        batch_end = min(batch_start + args.batch_size - 1, total - 1)

        # Write a temporary bitstring-file
        with tempfile.NamedTemporaryFile('w', delete=False) as tmp:
            for idx in range(batch_start, batch_end + 1):
                tmp.write(f"{idx} {valid[idx]}\n")
            tmp_path = tmp.name

        print(f"→ Running batch {batch_start//args.batch_size}: indices {batch_start}..{batch_end}")

        # Call the runner
        subprocess.run([
            sys.executable, args.runner_script,
            '--bitstring-file', tmp_path,
            '--time_to_run',     str(args.time),
        ], check=True)

        # Clean up
        os.remove(tmp_path)

if __name__ == '__main__':
    main()
