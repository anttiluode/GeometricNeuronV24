#!/usr/bin/env python3
"""Gate 7: can finite addressed delay history replace perfect HISTORY_REPLAY?

Gate 6C found an architectural fork:

- LOCAL_SAME_FIELD has a fast-write penalty.
- HISTORY_REPLAY stores every addressed pulse equation and removes that penalty.

The Takens/efference-copy discussion suggests an intermediate object: do not keep
perfect unbounded bookkeeping, but retain a finite temporal window of

    (probe address, returned scalar pulse)

and reconstruct only from that window.

At tau_write=0 (the hard fast-write point), this audit sweeps the history window
K.  K=1 is nearly memoryless; large K approaches Gate 6C's full replay.

A paired attacker keeps the scalar pulse history but deliberately attaches each
pulse to the wrong same-scale lens address.  This tests whether the intervention
/efference address is essential rather than delay memory alone.

This is a finite addressed-history estimator.  It is inspired by forced delay
embedding but is NOT a claim that classical Takens theorem applies to this toy
or that dendrites implement this exact pseudoinverse.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gate6c_write_timescale import (
    AUDIT_SEEDS,
    HOME_NOISE_SIGMA,
    MAX_SEARCH_PROBES,
    N,
    ON_STEPS,
    PAID_NOISE_SIGMA,
    PATCH_AMPLITUDE,
    POSTERIOR_STOP,
    SURPRISE_THRESHOLD,
    Lens,
    aggregate,
    make_lenses,
    make_templates,
    partition_information_bits,
    read,
    solve_history,
    trial as gate6c_trial,
)

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = (1, 2, 3, 4, 6, 8, 12, 16, 32, 64)


def wrong_same_scale_mask(lens: Lens, lenses: list[Lens]) -> np.ndarray:
    peers = [candidate for candidate in lenses if candidate.scale == lens.scale]
    i = next(i for i, candidate in enumerate(peers) if candidate.name == lens.name)
    return peers[(i + 1) % len(peers)].mask.copy()


def window_write(
    base: np.ndarray,
    memory: np.ndarray,
    lens: Lens,
    residual: float,
    history_masks: list[np.ndarray],
    history_values: list[float],
    lenses: list[Lens],
    window: int,
    scramble_address: bool,
) -> None:
    # Convert the residual into an absolute measurement before updating memory.
    absolute = float(np.sum(memory * lens.mask)) + float(residual)
    mask = (
        wrong_same_scale_mask(lens, lenses)
        if scramble_address
        else lens.mask.copy()
    )
    history_masks.append(mask)
    history_values.append(absolute)

    if len(history_masks) > window:
        del history_masks[:-window]
        del history_values[:-window]

    # tau=0 in Gate 7, so the finite-window estimate is applied immediately.
    memory[:] = solve_history(base, history_masks, history_values)


def search_window(
    world: np.ndarray,
    base: np.ndarray,
    memory: np.ndarray,
    templates: list[np.ndarray],
    lenses: list[Lens],
    rng: np.random.Generator,
    history_masks: list[np.ndarray],
    history_values: list[float],
    window: int,
    scramble_address: bool,
) -> dict:
    posterior = np.full(len(templates), 1.0 / len(templates), dtype=float)
    unused = list(range(len(lenses)))
    trace: list[dict] = []

    for _ in range(MAX_SEARCH_PROBES):
        scores = []
        means_by_lens = []
        for lens_index in unused:
            lens = lenses[lens_index]
            means = np.asarray(
                [read(base + template - memory, lens) for template in templates],
                dtype=float,
            )
            means_by_lens.append(means)
            scores.append(
                partition_information_bits(posterior, means, PAID_NOISE_SIGMA)
            )

        local_index = int(np.argmax(scores))
        lens_index = int(unused[local_index])
        lens = lenses[lens_index]
        means = means_by_lens[local_index]

        observed = read(world - memory, lens) + float(
            rng.normal(scale=PAID_NOISE_SIGMA)
        )
        logp = (
            np.log(np.maximum(posterior, 1e-300))
            - 0.5 * ((observed - means) / PAID_NOISE_SIGMA) ** 2
        )
        logp -= float(np.max(logp))
        posterior = np.exp(logp)
        posterior /= float(np.sum(posterior))

        trace.append(
            {
                "lens": lens.name,
                "scale": int(lens.scale),
                "observed_residual": float(observed),
                "posterior_max": float(np.max(posterior)),
            }
        )

        window_write(
            base,
            memory,
            lens,
            observed,
            history_masks,
            history_values,
            lenses,
            window,
            scramble_address,
        )
        unused.remove(lens_index)

        if float(np.max(posterior)) >= POSTERIOR_STOP:
            break

    return {
        "identity": int(np.argmax(posterior)),
        "confidence": float(np.max(posterior)),
        "probes": int(len(trace)),
        "trace": trace,
    }


def trial_window(
    identity: int,
    seed: int,
    window: int,
    *,
    scramble_address: bool,
) -> dict:
    base = np.zeros((N, N), dtype=float)
    templates = make_templates()
    lenses = make_lenses()
    world = base + templates[identity]
    memory = base.copy()

    offset = 700_001 if not scramble_address else 900_001
    rng = np.random.default_rng(seed * 1009 + identity * 53 + offset + window * 7)

    history_masks: list[np.ndarray] = []
    history_values: list[float] = []
    remembered_identity: int | None = None

    paid_probes = 0
    repeated_paid_steps = 0
    positive_searches = 0
    correct_searches = 0
    negative_home_triggers = 0
    scale_sequences: list[list[int]] = []

    for step in range(ON_STEPS):
        home_residual = float(np.mean(world - memory)) + float(
            rng.normal(scale=HOME_NOISE_SIGMA)
        )
        if abs(home_residual) < SURPRISE_THRESHOLD:
            continue

        if step > 0:
            repeated_paid_steps += 1

        if home_residual > 0:
            positive_searches += 1
            result = search_window(
                world,
                base,
                memory,
                templates,
                lenses,
                rng,
                history_masks,
                history_values,
                window,
                scramble_address,
            )
            paid_probes += int(result["probes"])
            scale_sequences.append(
                [int(row["scale"]) for row in result["trace"]]
            )
            if result["confidence"] >= POSTERIOR_STOP:
                remembered_identity = int(result["identity"])
            correct_searches += int(
                result["confidence"] >= POSTERIOR_STOP
                and int(result["identity"]) == identity
            )
        else:
            negative_home_triggers += 1
            fine_identity = 0 if remembered_identity is None else remembered_identity
            lens = next(
                candidate
                for candidate in lenses
                if candidate.scale == 4
                and candidate.name == f"s4_x{4 * fine_identity}"
            )
            observed = read(world - memory, lens) + float(
                rng.normal(scale=PAID_NOISE_SIGMA)
            )
            paid_probes += 1
            if observed < -0.5 * PATCH_AMPLITUDE:
                window_write(
                    base,
                    memory,
                    lens,
                    observed,
                    history_masks,
                    history_values,
                    lenses,
                    window,
                    scramble_address,
                )

    return {
        "paid_probes": int(paid_probes),
        "repeated_paid_steps": int(repeated_paid_steps),
        "positive_searches": int(positive_searches),
        "correct_searches": int(correct_searches),
        "negative_home_triggers": int(negative_home_triggers),
        "final_mse": float(np.mean((world - memory) ** 2)),
        "final_home_residual": float(np.mean(world - memory)),
        "scale_sequences": scale_sequences,
    }


def run_window(window: int, scramble: bool) -> dict:
    rows = [
        trial_window(identity, seed, window, scramble_address=scramble)
        for seed in AUDIT_SEEDS
        for identity in range(8)
    ]
    out = aggregate(rows)
    out["window"] = int(window)
    out["scramble_address"] = bool(scramble)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "gate7",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    local_rows = [
        gate6c_trial(identity, 0.0, seed, "LOCAL_SAME_FIELD")
        for seed in AUDIT_SEEDS
        for identity in range(8)
    ]
    full_rows = [
        gate6c_trial(identity, 0.0, seed, "HISTORY_REPLAY")
        for seed in AUDIT_SEEDS
        for identity in range(8)
    ]
    local = aggregate(local_rows)
    full = aggregate(full_rows)

    correct = [run_window(k, False) for k in WINDOWS]
    scrambled = [run_window(k, True) for k in WINDOWS]

    best = min(correct, key=lambda row: row["paid_probes_mean"])
    summary = {
        "schema": "geometric-neuron/gate7-finite-addressed-history-v1",
        "question": (
            "At instantaneous same-field WRITE, how much finite (address,pulse) history is needed to approach the full HISTORY_REPLAY attacker?"
        ),
        "baseline_local_same_field": local,
        "baseline_full_history_replay": full,
        "finite_correct_address": correct,
        "finite_scrambled_address": scrambled,
        "best_finite_window": best,
        "claim_boundary": (
            "This is finite addressed linear measurement replay, not a proof of Takens embedding, not a dendritic implementation, and not evidence for AIS periodicity as a delay line. The address-scramble arm tests the importance of retaining which intervention produced each scalar."
        ),
    }

    path = args.output_dir / "gate7_summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
