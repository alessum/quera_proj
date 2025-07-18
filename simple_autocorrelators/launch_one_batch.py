#!/usr/bin/env python3
import argparse, itertools, os, subprocess, sys, tempfile

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
    print(f"[DEBUG] mid_site={args.mid_site} (must be < {N})")
    
    valid = [
        ''.join(bits)
        for bits in itertools.product('01', repeat=N)
        if bits[args.mid_site] == '1'
    ]
    
    print(f"[DEBUG] Total valid bitstrings: {len(valid)}")
    print(f"[DEBUG] First few valid bitstrings: {valid[:3]}")

    start = args.batch_index * args.batch_size
    end   = min(start + args.batch_size - 1, len(valid) - 1)
    
    print(f"[DEBUG] Batch {args.batch_index}: indices {start}..{end} ({end-start+1} states)")

    # dump this batch to a temp file
    with tempfile.NamedTemporaryFile('w', delete=False) as tmp:
        for idx in range(start, end+1):
            tmp.write(f"{idx} {valid[idx]}\n")
        tmp_path = tmp.name
    
    print(f"[DEBUG] Generated bitstring length = {len(valid[0])}")
    print(f"[DEBUG] Temp file: {tmp_path}")
    assert args.mid_site < N, f"[DEBUG] mid_site ({args.mid_site}) must be less than N_sites ({N})"

    print(f"[DEBUG] Calling runner with: {args.runner_script}")
    subprocess.run([
        sys.executable, args.runner_script,
        '--bitstring-file', tmp_path,
        '--time_to_run',     str(args.time),
        '--Lx',              str(args.Lx),
        '--Ly',              str(args.Ly),
    ], check=True)

    os.remove(tmp_path)
    print(f"[DEBUG] === launch_one_batch.py END ===")


if __name__=='__main__':
    main()
