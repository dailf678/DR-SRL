#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
for method in sac-rfsi-dr-s sac-fsi-dr sac-fsi-a sac-lag-d; do
    python "$script_dir/train_rl.py" --alg "$method" "$@"
done
