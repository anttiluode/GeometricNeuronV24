#!/usr/bin/env python3
"""Gate 9R: remove the 'current address summarizes history' shortcut.

Gate 9's TAP1 model unexpectedly matched full replay.  Inspection gives a clean
confound: under the adaptive 16->8->4 policy, the *current fine address* is chosen
from previous observations, so the last action itself encodes much of the search
history.  A memoryless decoder can therefore look temporal when it is actually
reading the policy's state through its chosen address.

Gate 9R uses a random-order scan of all eight fine 4x4 lenses.  The permutation
is independent of the hidden anomaly and varies every trial.  Thus:

- the current address no longer summarizes previous evidence;
- delay position alone no longer identifies which spatial address was sampled;
- the estimator still has to account for its own previous writes because each
  returned pulse is measured against the current internal estimate.

We train the same linear decoders to imitate full-history replay and test them
closed-loop.  The primary endpoint is final hidden-identity accuracy after the
8-probe scan; MSE is secondary.

This is still an engineered system-identification toy, not a biological model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gate6c_write_timescale import (
    AUDIT_SEEDS,
    N,
    PAID_NOISE_SIGMA,
    make_lenses,
    make_templates,
    read,
    solve_history,
)
from gate9_temporal_filter_bank import (
    RidgeDecoder,
    event_vector,
    make_state,
)

ROOT = Path(__file__).resolve().parents[1]


def fine_lenses():
    return [lens for lens in make_lenses() if lens.scale == 4]


def identity_from_estimate(memory: np.ndarray, templates: list[np.ndarray]) -> int:
    # Nearest template under MSE.  This readout is fixed and receives no trial
    # identity or scan-order information.
    return int(np.argmin([np.mean((memory - t) ** 2) for t in templates]))


def generate_samples(kind: str, address_mode: str, seeds=range(101, 181)):
    lenses = fine_lenses()
    templates = make_templates()
    width = 2 if address_mode == "scalar_only" else 2 * len(lenses)
    X, Y = [], []

    for seed in seeds:
        for identity in range(8):
            rng = np.random.default_rng(10_000_019 + seed * 1009 + identity * 43)
            order = rng.permutation(len(lenses))
            base = np.zeros((N, N), dtype=float)
            world = templates[identity].copy()
            memory = base.copy()
            H, values = [], []
            state = make_state(kind, width)

            for idx in order:
                lens = lenses[int(idx)]
                residual = read(world - memory, lens) + float(
                    rng.normal(scale=PAID_NOISE_SIGMA)
                )
                absolute = float(np.sum(memory * lens.mask)) + residual
                H.append(lens.mask.copy())
                values.append(absolute)
                memory = solve_history(base, H, values)
                state.push(event_vector(
                    lens, absolute, lenses, address_mode=address_mode
                ))
                X.append(state.feature().copy())
                Y.append(memory.reshape(-1).copy())

    return np.asarray(X), np.asarray(Y)


def train(kind: str, address_mode: str):
    X, Y = generate_samples(kind, address_mode)
    return RidgeDecoder(ridge=1e-4).fit(X, Y), list(X.shape)


def closed_loop_trial(
    identity: int,
    seed: int,
    kind: str,
    decoder: RidgeDecoder,
    address_mode: str,
) -> dict:
    lenses = fine_lenses()
    templates = make_templates()
    width = 2 if address_mode == "scalar_only" else 2 * len(lenses)
    state = make_state(kind, width)
    rng = np.random.default_rng(11_000_033 + seed * 1009 + identity * 47)
    order = rng.permutation(len(lenses))
    world = templates[identity].copy()
    memory = np.zeros((N, N), dtype=float)

    trace = []
    for idx in order:
        lens = lenses[int(idx)]
        residual = read(world - memory, lens) + float(
            rng.normal(scale=PAID_NOISE_SIGMA)
        )
        absolute = float(np.sum(memory * lens.mask)) + residual
        state.push(event_vector(
            lens, absolute, lenses, address_mode=address_mode
        ))
        memory = decoder.predict(state.feature()).reshape(N, N)
        trace.append(lens.name)

    pred = identity_from_estimate(memory, templates)
    return {
        "correct": int(pred == identity),
        "prediction": int(pred),
        "identity": int(identity),
        "mse": float(np.mean((memory - world) ** 2)),
        "order": trace,
    }


def evaluate(kind, decoder, address_mode):
    rows = [
        closed_loop_trial(identity, seed, kind, decoder, address_mode)
        for seed in AUDIT_SEEDS
        for identity in range(8)
    ]
    return {
        "kind": kind,
        "address_mode": address_mode,
        "trials": len(rows),
        "identity_accuracy": float(np.mean([r["correct"] for r in rows])),
        "mse_mean": float(np.mean([r["mse"] for r in rows])),
        "mse_median": float(np.median([r["mse"] for r in rows])),
    }


def exact_full_replay_baseline():
    lenses = fine_lenses()
    templates = make_templates()
    rows = []
    for seed in AUDIT_SEEDS:
        for identity in range(8):
            rng = np.random.default_rng(12_000_037 + seed * 1009 + identity * 47)
            order = rng.permutation(len(lenses))
            world = templates[identity].copy()
            memory = np.zeros((N, N), dtype=float)
            H, values = [], []
            for idx in order:
                lens = lenses[int(idx)]
                residual = read(world - memory, lens) + float(
                    rng.normal(scale=PAID_NOISE_SIGMA)
                )
                absolute = float(np.sum(memory * lens.mask)) + residual
                H.append(lens.mask.copy())
                values.append(absolute)
                memory = solve_history(np.zeros_like(memory), H, values)
            rows.append({
                "correct": identity_from_estimate(memory, templates) == identity,
                "mse": float(np.mean((memory - world) ** 2)),
            })
    return {
        "identity_accuracy": float(np.mean([r["correct"] for r in rows])),
        "mse_mean": float(np.mean([r["mse"] for r in rows])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results" / "gate9r"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    models = {}
    shapes = {}
    for kind in ("tap1", "tap2", "tap3", "leaky4"):
        models[kind], shapes[kind] = train(kind, "correct")
    scalar, shapes["leaky4_scalar_only"] = train("leaky4", "scalar_only")

    summary = {
        "schema": "geometric-neuron/gate9r-random-scan-history-v1",
        "question": (
            "When adaptive-policy address leakage is removed, does temporal history of correctly addressed self-generated pulses become necessary?"
        ),
        "scan": "random permutation of all eight fine 4x4 lenses; order independent of hidden identity",
        "training_shapes": shapes,
        "full_history_replay": exact_full_replay_baseline(),
        "conditions": [
            evaluate("tap1", models["tap1"], "correct"),
            evaluate("tap2", models["tap2"], "correct"),
            evaluate("tap3", models["tap3"], "correct"),
            evaluate("leaky4", models["leaky4"], "correct"),
            evaluate("leaky4", scalar, "scalar_only"),
        ],
        "claim_boundary": (
            "This gate removes one policy-state shortcut. It still uses a trained linear decoder and synthetic fine-lens scans. A temporal advantage supports finite history as a useful state variable in this toy, not Takens theorem or a dendritic implementation."
        ),
    }
    path = args.output_dir / "gate9r_summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
