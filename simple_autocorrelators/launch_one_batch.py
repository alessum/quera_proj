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

    # total sites and all valid bitstrings:
    N = args.Lx * args.Ly
    print(f"[DEBUG] Lx={args.Lx}, Ly={args.Ly} → N_sites={N}")
    valid = [
        ''.join(bits)
        for bits in itertools.product('01', repeat=N)
        if bits[args.mid_site] == '1'
    ]

    start = args.batch_index * args.batch_size
    end   = min(start + args.batch_size - 1, len(valid) - 1)

    # dump this batch to a temp file
    with tempfile.NamedTemporaryFile('w', delete=False) as tmp:
        for idx in range(start, end+1):
            tmp.write(f"{idx} {valid[idx]}\n")
        tmp_path = tmp.name
    
    print(f"[DEBUG] Generated bitstring length = {len(valid[0])}")
    assert args.mid_site < N, "[DEBUG] mid_site must be less than N_sites"

    print(f"[batch {args.batch_index}] indices {start}..{end}")
    subprocess.run([
        sys.executable, args.runner_script,
        '--bitstring-file', tmp_path,
        '--time_to_run',     str(args.time),
        '--Lx',              str(args.Lx),
        '--Ly',              str(args.Ly),
    ], check=True)

    os.remove(tmp_path)


if __name__=='__main__':
    main()
