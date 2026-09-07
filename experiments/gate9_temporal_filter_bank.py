#!/usr/bin/env python3
"""Gate 9: can a temporal filter bank approximate the protected reference state?

Gate 8 showed that a rank-3 streaming intervention subspace can match perfect
HISTORY_REPLAY without keeping a chronological ledger.  That update uses exact
orthogonalization, which is useful mathematics but not a plausible cable model.

Gate 9 tests the older dendrite / cerebellar-adaptive-filter intuition in the
narrowest engineering form:

    addressed scalar pulse
        -> bank of delayed / leaky traces
        -> linear decoder
        -> current reference field

The decoder is trained only to imitate the full-history reference on independent
self-generated measurement sequences.  At test time it runs closed-loop in the
same eight-step persistent-anomaly task as Gate 6C.

Conditions:
- TAP1/TAP2/TAP3: exact event-delay taps of the last K addressed pulses.
- LEAKY4: four exponentially filtered traces with different decay rates.
- LEAKY4_SCALAR_ONLY: same temporal bank but address identity is removed.
- LEAKY4_WRONG_ADDRESS: correct model, but test pulses are tagged with the wrong
  same-scale address.

This is a learned approximation / expressivity audit.  A positive result would
show that a small temporal basis can approximate the computational reference
identified by Gates 7-8.  It would not show that dendrites implement the trained
decoder, that Takens' theorem applies, or that AIS periodicity is involved.
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
    aggregate,
    make_lenses,
    make_templates,
    partition_information_bits,
    read,
    solve_history,
    trial as gate6c_trial,
)
from gate7_forced_history import wrong_same_scale_mask

ROOT = Path(__file__).resolve().parents[1]


def event_vector(lens, absolute: float, lenses, *, address_mode: str) -> np.ndarray:
    """Address-specific value + occupancy channels, or scalar-only attacker."""
    if address_mode == "scalar_only":
        return np.asarray([float(absolute), 1.0], dtype=float)

    use_lens = lens
    if address_mode == "wrong":
        wrong = wrong_same_scale_mask(lens, lenses)
        # Recover the corresponding lens object solely for one-hot tagging.
        use_lens = next(x for x in lenses if np.array_equal(x.mask, wrong))
    elif address_mode != "correct":
        raise ValueError(address_mode)

    idx = next(i for i, x in enumerate(lenses) if x.name == use_lens.name)
    n = len(lenses)
    out = np.zeros(2 * n, dtype=float)
    out[idx] = float(absolute)
    out[n + idx] = 1.0
    return out


class TapState:
    def __init__(self, width: int, taps: int):
        self.width = int(width)
        self.taps = int(taps)
        self.rows = [np.zeros(self.width, dtype=float) for _ in range(self.taps)]

    def push(self, event: np.ndarray) -> None:
        self.rows = [np.asarray(event, dtype=float).copy()] + self.rows[:-1]

    def feature(self) -> np.ndarray:
        return np.concatenate(self.rows)


class LeakyState:
    def __init__(self, width: int, decays=(0.0, 0.5, 0.8, 0.95)):
        self.decays = tuple(float(d) for d in decays)
        self.rows = [np.zeros(int(width), dtype=float) for _ in self.decays]

    def push(self, event: np.ndarray) -> None:
        event = np.asarray(event, dtype=float)
        for i, decay in enumerate(self.decays):
            self.rows[i] = decay * self.rows[i] + event

    def feature(self) -> np.ndarray:
        return np.concatenate(self.rows)


def make_state(kind: str, width: int):
    if kind.startswith("tap"):
        return TapState(width, int(kind[3:]))
    if kind == "leaky4":
        return LeakyState(width)
    raise ValueError(kind)


class RidgeDecoder:
    def __init__(self, ridge=1e-4):
        self.ridge = float(ridge)
        self.mean = None
        self.scale = None
        self.beta = None

    def fit(self, X: np.ndarray, Y: np.ndarray):
        self.mean = X.mean(axis=0)
        self.scale = X.std(axis=0)
        self.scale[self.scale < 1e-10] = 1.0
        Z = (X - self.mean) / self.scale
        Z = np.column_stack([np.ones(len(Z)), Z])
        reg = np.eye(Z.shape[1]) * self.ridge
        reg[0, 0] = 0.0
        self.beta = np.linalg.solve(Z.T @ Z + reg, Z.T @ Y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = (np.asarray(x) - self.mean) / self.scale
        z = np.concatenate([[1.0], z])
        return z @ self.beta


def training_samples(kind: str, address_mode: str, train_seeds=range(101, 141)):
    lenses = make_lenses()
    templates = make_templates()
    width = 2 if address_mode == "scalar_only" else 2 * len(lenses)
    X, Y = [], []

    for seed in train_seeds:
        for identity in range(8):
            rng = np.random.default_rng(9_000_001 + seed * 101 + identity)
            base = np.zeros((N, N), dtype=float)
            world = base + templates[identity]
            memory = base.copy()
            posterior = np.full(8, 1.0 / 8.0)
            unused = list(range(len(lenses)))
            hist_masks, hist_values = [], []
            state = make_state(kind, width)

            for _ in range(MAX_SEARCH_PROBES):
                scores, means_by = [], []
                for lens_index in unused:
                    lens = lenses[lens_index]
                    means = np.asarray([
                        read(base + template - memory, lens)
                        for template in templates
                    ])
                    means_by.append(means)
                    scores.append(partition_information_bits(
                        posterior, means, PAID_NOISE_SIGMA
                    ))
                j = int(np.argmax(scores))
                lens_index = int(unused[j])
                lens = lenses[lens_index]
                means = means_by[j]
                observed = read(world - memory, lens) + float(
                    rng.normal(scale=PAID_NOISE_SIGMA)
                )
                absolute = float(np.sum(memory * lens.mask)) + observed

                logp = np.log(np.maximum(posterior, 1e-300)) - 0.5 * (
                    (observed - means) / PAID_NOISE_SIGMA
                ) ** 2
                logp -= np.max(logp)
                posterior = np.exp(logp)
                posterior /= np.sum(posterior)

                hist_masks.append(lens.mask.copy())
                hist_values.append(float(absolute))
                memory = solve_history(base, hist_masks, hist_values)

                state.push(event_vector(
                    lens, absolute, lenses, address_mode=address_mode
                ))
                X.append(state.feature().copy())
                Y.append(memory.reshape(-1).copy())
                unused.remove(lens_index)
                if float(np.max(posterior)) >= POSTERIOR_STOP:
                    break

    return np.asarray(X), np.asarray(Y)


def train_decoder(kind: str, address_mode: str):
    X, Y = training_samples(kind, address_mode)
    return RidgeDecoder().fit(X, Y), X.shape


def trial(
    identity: int,
    seed: int,
    *,
    kind: str,
    decoder: RidgeDecoder,
    address_mode: str,
) -> dict:
    base = np.zeros((N, N), dtype=float)
    templates = make_templates()
    lenses = make_lenses()
    world = base + templates[identity]
    memory = base.copy()
    width = 2 if address_mode == "scalar_only" else 2 * len(lenses)
    # Wrong-address evaluation uses the correct-address feature width.
    if address_mode == "wrong":
        width = 2 * len(lenses)
    state = make_state(kind, width)
    rng = np.random.default_rng(9_500_003 + seed * 1009 + identity * 53)

    remembered_identity = None
    paid_probes = repeated_paid_steps = positive_searches = 0
    correct_searches = negative_home_triggers = 0
    scale_sequences = []

    def absorb(lens, observed):
        nonlocal memory
        absolute = float(np.sum(memory * lens.mask)) + float(observed)
        state.push(event_vector(
            lens, absolute, lenses, address_mode=address_mode
        ))
        pred = decoder.predict(state.feature()).reshape(N, N)
        # Decoder can make tiny off-manifold values; keep them finite but do not
        # clip sign or force the known patch geometry.
        memory = np.nan_to_num(pred, nan=0.0, posinf=2.0, neginf=-2.0)

    for step in range(ON_STEPS):
        home = float(np.mean(world - memory)) + float(
            rng.normal(scale=HOME_NOISE_SIGMA)
        )
        if abs(home) < SURPRISE_THRESHOLD:
            continue
        if step > 0:
            repeated_paid_steps += 1

        if home > 0:
            positive_searches += 1
            posterior = np.full(8, 1.0 / 8.0)
            unused = list(range(len(lenses)))
            trace = []
            for _ in range(MAX_SEARCH_PROBES):
                scores, means_by = [], []
                for lens_index in unused:
                    lens = lenses[lens_index]
                    means = np.asarray([
                        read(base + template - memory, lens)
                        for template in templates
                    ])
                    means_by.append(means)
                    scores.append(partition_information_bits(
                        posterior, means, PAID_NOISE_SIGMA
                    ))
                j = int(np.argmax(scores))
                lens_index = int(unused[j])
                lens = lenses[lens_index]
                means = means_by[j]
                observed = read(world - memory, lens) + float(
                    rng.normal(scale=PAID_NOISE_SIGMA)
                )
                logp = np.log(np.maximum(posterior, 1e-300)) - 0.5 * (
                    (observed - means) / PAID_NOISE_SIGMA
                ) ** 2
                logp -= np.max(logp)
                posterior = np.exp(logp)
                posterior /= np.sum(posterior)
                absorb(lens, observed)
                trace.append(int(lens.scale))
                unused.remove(lens_index)
                if float(np.max(posterior)) >= POSTERIOR_STOP:
                    break

            paid_probes += len(trace)
            scale_sequences.append(trace)
            if float(np.max(posterior)) >= POSTERIOR_STOP:
                remembered_identity = int(np.argmax(posterior))
            correct_searches += int(
                float(np.max(posterior)) >= POSTERIOR_STOP
                and int(np.argmax(posterior)) == identity
            )
        else:
            negative_home_triggers += 1
            fine = 0 if remembered_identity is None else remembered_identity
            lens = next(
                x for x in lenses
                if x.scale == 4 and x.name == f"s4_x{4 * fine}"
            )
            observed = read(world - memory, lens) + float(
                rng.normal(scale=PAID_NOISE_SIGMA)
            )
            paid_probes += 1
            if observed < -0.5 * PATCH_AMPLITUDE:
                absorb(lens, observed)

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


def evaluate_condition(kind, decoder, address_mode):
    rows = [
        trial(identity, seed, kind=kind, decoder=decoder, address_mode=address_mode)
        for seed in AUDIT_SEEDS for identity in range(8)
    ]
    out = aggregate(rows)
    out.update({"kind": kind, "address_mode": address_mode})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results" / "gate9"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    trained = {}
    train_shapes = {}
    for kind in ("tap1", "tap2", "tap3", "leaky4"):
        model, shape = train_decoder(kind, "correct")
        trained[kind] = model
        train_shapes[kind] = list(shape)
    scalar_model, scalar_shape = train_decoder("leaky4", "scalar_only")

    local = aggregate([
        gate6c_trial(identity, 0.0, seed, "LOCAL_SAME_FIELD")
        for seed in AUDIT_SEEDS for identity in range(8)
    ])
    full = aggregate([
        gate6c_trial(identity, 0.0, seed, "HISTORY_REPLAY")
        for seed in AUDIT_SEEDS for identity in range(8)
    ])

    conditions = [
        evaluate_condition("tap1", trained["tap1"], "correct"),
        evaluate_condition("tap2", trained["tap2"], "correct"),
        evaluate_condition("tap3", trained["tap3"], "correct"),
        evaluate_condition("leaky4", trained["leaky4"], "correct"),
        evaluate_condition("leaky4", scalar_model, "scalar_only"),
        evaluate_condition("leaky4", trained["leaky4"], "wrong"),
    ]

    summary = {
        "schema": "geometric-neuron/gate9-temporal-filter-bank-v1",
        "question": (
            "Can a small delay/leaky basis driven by addressed scalar pulses learn to approximate the reference state that exact finite bookkeeping provided?"
        ),
        "training_shapes": {**train_shapes, "leaky4_scalar_only": list(scalar_shape)},
        "baseline_local_same_field": local,
        "baseline_full_history_replay": full,
        "conditions": conditions,
        "claim_boundary": (
            "The filter bank and linear decoder are an engineering approximation trained against full replay. A positive result establishes representational sufficiency in this toy, not a biological cable implementation, Takens theorem, cerebellar identity, or AIS grating."
        ),
    }
    path = args.output_dir / "gate9_summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
