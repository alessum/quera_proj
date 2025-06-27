# scripts/launch_simulations.sh
#!/usr/bin/env bash
set -euo pipefail

# Inputs
# Usage: launch_simulations.sh <start_index> <count> <theta> <T_run> [mid_site] [Lx] [Ly]
start_index=${1:-0}
count=${2:-0}
T_run=${4:-0}
mid_site_arg=${5:-}
Lx_arg=${6:-}
Ly_arg=${7:-}

# Defaults
Lx=${Lx_arg:-3}
Ly=${Ly_arg:-3}
N_sites=$((Lx * Ly))
# Default mid_site to center if not provided
mid_site=${mid_site_arg:-$((N_sites/2))}

batch_size=50

# Prepare base output directory
base_out_dir="results/L${Lx}/T${T_run}/${mid_site}/"
mkdir -p "$base_out_dir"

echo "Processing all bitstrings for Lx=$Lx, Ly=$Ly (mid_site=$mid_site) in batches of $batch_size"

# Total valid combinations where bit d == '1'
total=$((1 << (N_sites-1)))

# Enumerate in batches
for (( batch_start=0; batch_start<total; batch_start+=batch_size )); do
  batch_end=$(( batch_start + batch_size - 1 ))
  if (( batch_end >= total )); then batch_end=$(( total - 1 )); fi

  # Create a temp file listing circuit_id + bitstring for this batch
  bs_file=$(mktemp)
  trap 'rm -f "$bs_file"' EXIT
  echo "Generating bitstrings indices $batch_start..$batch_end"
  python3 - <<EOF > "$bs_file"
import itertools
N = $N_sites
d = $mid_site
valid = [''.join(bits) for bits in itertools.product('01', repeat=N) if bits[d]=='1']
for idx, bs in enumerate(valid[$batch_start:$((batch_end+1))], start=$batch_start):
    print(idx, bs)
EOF

  # Run simulations for this batch
  python runner_ac_data.py \
    --bitstring-file "$bs_file" \
    --time_to_run "$T_run" \
    --output-dir "$base_out_dir"

  rm -f "$bs_file"
  trap - EXIT

done