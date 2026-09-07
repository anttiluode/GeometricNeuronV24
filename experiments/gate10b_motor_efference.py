#!/usr/bin/env python3
"""Gate 10B: isolate efference copy from active sensor selection.

Gate 10 let action do two things: kick the hidden phase and select the scalar
sensor direction.  Therefore a skeptic can say that the 'efference' benefit is
partly just knowing which sensor was used.

Gate 10B removes that explanation.  All four actions are observed through the
same fixed scalar sensor.  Action identity now matters ONLY because each action
causes a different known phase kick before the return is measured.

The observer still sees a scalar time series from one sensor and a copy of the
motor-like actions that perturbed the hidden oscillator.  If delayed history plus
that action copy predicts return and exposes hidden phase shocks, the result is a
cleaner corollary-discharge analogue.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import gate10_controlled_delay_observer as g10

ROOT = Path(__file__).resolve().parents[1]


def aggregate(runs):
    return {
        "mse": float(np.mean([r["mse"] for r in runs])),
        "mae": float(np.mean([r["mae"] for r in runs])),
        "omega_mae": float(np.mean([r["omega_mae"] for r in runs])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results" / "gate10b"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Critical intervention: sensor direction is identical for every action.
    # Actions remain different only through their physical phase kicks KAPPA.
    g10.PHI[:] = 0.0

    clean = [g10.simulate(seed, shocks=False) for seed in range(g10.CFG.seeds)]
    shocked = [
        g10.simulate(300 + seed, shocks=True) for seed in range(g10.CFG.seeds)
    ]

    sweep = []
    for k in g10.WINDOWS:
        sweep.append({
            "window": int(k),
            **aggregate([g10.prediction_run(traj, k) for traj in clean]),
        })
    best = min(sweep, key=lambda row: row["mse"])
    k = int(best["window"])

    wrong = aggregate([
        g10.prediction_run(traj, k, wrong_history_address=True)
        for traj in clean
    ])
    no_next = aggregate([
        g10.prediction_run(traj, k, know_next_action=False)
        for traj in clean
    ])

    shock_runs = [g10.prediction_run(traj, k) for traj in shocked]
    labels = np.concatenate([r["labels"] for r in shock_runs])
    innovation = np.concatenate([r["innovation"] for r in shock_runs])
    raw = np.concatenate([r["raw_change"] for r in shock_runs])
    wrong_shock_runs = [
        g10.prediction_run(traj, k, wrong_history_address=True)
        for traj in shocked
    ]
    wrong_innovation = np.concatenate([
        r["innovation"] for r in wrong_shock_runs
    ])

    summary = {
        "schema": "geometric-neuron/gate10b-motor-efference-v1",
        "question": (
            "With one fixed sensor, does history plus a copy of self-generated motor-like kicks still reconstruct hidden state and improve innovation detection?"
        ),
        "critical_control": (
            "PHI is identical for all actions; action identity affects only the known hidden-state kick KAPPA before sensory return."
        ),
        "clean_prediction_by_history": sweep,
        "best_window": k,
        "attackers_at_best_window": {
            "wrong_past_motor_copy": wrong,
            "no_next_motor_efference_copy": no_next,
        },
        "hidden_phase_shock": {
            "n_shocks": int(labels.sum()),
            "n_clean": int((labels == 0).sum()),
            "auc_raw_abs_change": g10.auc(raw, labels),
            "auc_abs_innovation": g10.auc(innovation, labels),
            "auc_abs_innovation_wrong_motor_copy": g10.auc(
                wrong_innovation, labels
            ),
        },
        "claim_boundary": (
            "This removes action-dependent sensor selection from Gate 10. It is still a synthetic oscillator with a known motor-effect model, not evidence that a specific neural circuit, dendrite, AIS or electric-fish circuit implements this estimator."
        ),
    }
    path = args.output_dir / "gate10b_summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
