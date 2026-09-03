#!/usr/bin/env python3
"""Robustness audit for Gate 3 across seeds 1..10."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from gate3_read_write import ROOT, run


def audit(output_dir: Path) -> dict:
    seeds = list(range(1, 11))
    with tempfile.TemporaryDirectory(prefix="gnv24-gate3-audit-") as temporary:
        temp_root = Path(temporary)
        runs = [run(seed, temp_root / f"seed_{seed}") for seed in seeds]

    a_active = np.asarray([item["gate3a"]["active"]["final_accuracy"] for item in runs])
    a_random = np.asarray([item["gate3a"]["random"]["final_accuracy"] for item in runs])
    b_active = np.asarray([item["gate3b"]["noise_aware"]["final_accuracy"] for item in runs])
    b_raw = np.asarray([item["gate3b"]["raw_variance"]["final_accuracy"] for item in runs])
    b_random = np.asarray([item["gate3b"]["random"]["final_accuracy"] for item in runs])
    pass_a = np.asarray([item["gate3a"]["passed"] for item in runs], dtype=bool)
    pass_b = np.asarray([item["gate3b"]["passed"] for item in runs], dtype=bool)

    robust = bool(np.all(pass_a) and np.all(pass_b))
    result = {
        "audit": "gate3_robustness",
        "seeds": seeds,
        "passed": robust,
        "classification": "READ_WRITE_MECHANISM_REPLICATES_10_OF_10" if robust else "GATE3_NOT_ROBUST",
        "gate3a": {
            "pass_count": int(np.sum(pass_a)),
            "active_accuracy_mean": float(np.mean(a_active)),
            "active_accuracy_min": float(np.min(a_active)),
            "random_accuracy_mean": float(np.mean(a_random)),
            "active_minus_random_min": float(np.min(a_active - a_random)),
        },
        "gate3b": {
            "pass_count": int(np.sum(pass_b)),
            "noise_aware_accuracy_mean": float(np.mean(b_active)),
            "noise_aware_accuracy_min": float(np.min(b_active)),
            "raw_variance_accuracy_mean": float(np.mean(b_raw)),
            "random_accuracy_mean": float(np.mean(b_random)),
            "noise_aware_minus_random_min": float(np.min(b_active - b_random)),
        },
        "note": "Gate 3A and 3B retain their locked thresholds on all ten new seeds.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "robustness.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "gate3_robustness")
    args = parser.parse_args()
    result = audit(args.output_dir)
    print("GeometricNeuronV24 Gate 3 robustness audit")
    print()
    print(f"Gate 3A passes:                       {result['gate3a']['pass_count']} / 10")
    print(f"  active accuracy mean/min:           {result['gate3a']['active_accuracy_mean']:.3f} / {result['gate3a']['active_accuracy_min']:.3f}")
    print(f"  random accuracy mean:               {result['gate3a']['random_accuracy_mean']:.3f}")
    print(f"Gate 3B passes:                       {result['gate3b']['pass_count']} / 10")
    print(f"  noise-aware accuracy mean/min:      {result['gate3b']['noise_aware_accuracy_mean']:.3f} / {result['gate3b']['noise_aware_accuracy_min']:.3f}")
    print(f"  raw variance / random mean:         {result['gate3b']['raw_variance_accuracy_mean']:.3f} / {result['gate3b']['random_accuracy_mean']:.3f}")
    print()
    print(result["classification"])
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
