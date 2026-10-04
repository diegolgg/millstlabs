#!/usr/bin/env bash
set -euo pipefail
run_root="${1:-runs/smoke}"
mkdir -p "$run_root"
millst --config configs/smoke.yaml warmstart --seed 11 --output "$run_root/warm.pt"
for method in iteration r_adult r_initial; do
  millst --config configs/smoke.yaml train --profile prosocial --method "$method" --seed 11 --warmstart "$run_root/warm.pt" --output "$run_root/$method"
done
millst --config configs/smoke.yaml analyze --root "$run_root" --output "$run_root/analysis.json"
