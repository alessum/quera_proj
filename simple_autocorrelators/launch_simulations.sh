#!/usr/bin/env bash
set -euo pipefail

# Inputs
start_index=$1
count=$2
theta=$3
T_run=$4
batch_size=50

# System parameters
Lx=3
Ly=3
N_SITES=$((Lx * Ly))
mid_site=4               # fixed central site index

# Prepare base output directory
base_out_dir="results/U1/T${T_run}/theta$(printf '%.2f' "$theta")"
mkdir -p "$base_out_dir"

# Determine end index
end_index=$((start_index + count - 1))
echo "Processing circuits $start_index..$end_index in batches of $batch_size"

# Total combinations with mid_site forced to '1'
# There are 2^(N_SITES-1) possible bitstrings
# Use lex order of integers skipping those with mid bit == 0
total=$((1 << (N_SITES - 1)))

# Iterate in batches
for (( batch_start=0; batch_start<total; batch_start+=batch_size )); do
  batch_end=$(( batch_start + batch_size - 1 ))
  if (( batch_end >= total )); then batch_end=$((total - 1)); fi

  # Generate bitstring file with sorted lex indices
  bs_file=$(mktemp)
  echo "Generating bitstrings indices $batch_start..$batch_end"
  python3 - <<EOF > "$bs_file"
import sys, itertools
N = $N_SITES
d = $mid_site
start, end = $batch_start, $batch_end
i = 0
for bits in itertools.product('01', repeat=N):
    if bits[d] != '1': continue
    if i < start or i > end:
        i += 1
        continue
    bs = ''.join(bits)
    print(i, bs)
    i += 1
if i <= end:
    print("Warning: fewer combinations than requested range", file=sys.stderr)
EOF

  # Run runner in batch mode
  python runnerU1_pure.py \
    --bitstring-file "$bs_file" \
    --theta_to_run "$theta" \
    --time_to_run "$T_run" \
    --output-dir "$base_out_dir"

  rm "$bs_file"
done
