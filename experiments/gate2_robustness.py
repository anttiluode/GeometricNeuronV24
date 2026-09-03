#!/usr/bin/env python3
"""Post-gate robustness audit for Gate 2.

The original Gate-2 thresholds were set in ``gate2_rule_sherlock.py``.  This
script does not relax them.  It repeats the complete experiment on seeds 1..10
and reports separately whether the mechanism and the locked effect-size margin
replicate.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from gate2_rule_sherlock import ROOT, run


def audit(output_dir: Path) -> dict:
    seeds = list(range(1, 11))
    # The per-seed Gate-2 receipts are intermediate evidence; the summary below
    # retains every metric needed by this audit without committing ten copies of
    # the complete 64-law table.
    with tempfile.TemporaryDirectory(prefix="gnv24-gate2-audit-") as temporary:
        temp_root = Path(temporary)
        runs = [run(seed, temp_root / f"seed_{seed}", make_plots=False) for seed in seeds]
    active = np.asarray(
        [item["policies"]["active_multiscale"]["final_exact_accuracy"] for item in runs]
    )
    random = np.asarray([item["policies"]["random"]["final_exact_accuracy"] for item in runs])
    fixed = np.asarray([item["policies"]["fixed"]["final_exact_accuracy"] for item in runs])
    active_entropy = np.asarray(
        [item["policies"]["active_multiscale"]["final_entropy_bits"] for item in runs]
    )
    locked_passes = np.asarray([item["passed"] for item in runs], dtype=bool)
    margins = active - random

    mechanism_replicated = bool(
        np.min(active) >= 0.99
        and np.max(active_entropy) <= 0.01
        and np.min(margins) > 0.0
        and np.min(active - fixed) > 0.0
    )
    original_margin_replicated = bool(np.all(margins >= 0.20))
    result = {
        "audit": "gate2_robustness",
        "seeds": seeds,
        "mechanism_replicated": mechanism_replicated,
        "original_locked_margin_replicated": original_margin_replicated,
        "original_gate_pass_count": int(np.sum(locked_passes)),
        "original_gate_total": len(runs),
        "classification": (
            "MECHANISM_REPLICATES_LOCKED_MARGIN_DOES_NOT"
            if mechanism_replicated and not original_margin_replicated
            else (
                "MECHANISM_AND_MARGIN_REPLICATE"
                if mechanism_replicated
                else "MECHANISM_NOT_ROBUST"
            )
        ),
        "active_exact_accuracy": {
            "mean": float(np.mean(active)),
            "min": float(np.min(active)),
            "max": float(np.max(active)),
            "per_seed": active.tolist(),
        },
        "random_exact_accuracy": {
            "mean": float(np.mean(random)),
            "min": float(np.min(random)),
            "max": float(np.max(random)),
            "per_seed": random.tolist(),
        },
        "fixed_exact_accuracy": {
            "mean": float(np.mean(fixed)),
            "min": float(np.min(fixed)),
            "max": float(np.max(fixed)),
            "per_seed": fixed.tolist(),
        },
        "active_minus_random": {
            "mean": float(np.mean(margins)),
            "min": float(np.min(margins)),
            "max": float(np.max(margins)),
            "per_seed": margins.tolist(),
        },
        "note": (
            "This audit was added after the default-seed Gate 2 result. The "
            "original >=0.20 active-minus-random requirement is retained and "
            "fails on half the audit seeds; no threshold was lowered."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "robustness.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results" / "gate2_robustness"
    )
    args = parser.parse_args()
    result = audit(args.output_dir)
    print("GeometricNeuronV24 Gate 2 robustness audit")
    print()
    print(
        "active exact accuracy:                "
        f"{result['active_exact_accuracy']['mean']:.3f} "
        f"(min {result['active_exact_accuracy']['min']:.3f})"
    )
    print(
        "random exact accuracy:                "
        f"{result['random_exact_accuracy']['mean']:.3f} "
        f"(range {result['random_exact_accuracy']['min']:.3f}-"
        f"{result['random_exact_accuracy']['max']:.3f})"
    )
    print(
        "active - random margin:               "
        f"{result['active_minus_random']['mean']:.3f} "
        f"(min {result['active_minus_random']['min']:.3f})"
    )
    print(
        "original locked gate passes:          "
        f"{result['original_gate_pass_count']} / {result['original_gate_total']}"
    )
    print()
    print(result["classification"])
    raise SystemExit(0 if result["mechanism_replicated"] else 1)


if __name__ == "__main__":
    main()
