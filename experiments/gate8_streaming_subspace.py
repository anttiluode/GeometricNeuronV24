#!/usr/bin/env python3
"""Gate 8: replace the pulse ledger with a streaming reference subspace.

Gate 7 showed that the last three correctly addressed measurements are enough to
match the full HISTORY_REPLAY estimator in the locked 16->8->4 search.  But Gate
7 still stores an explicit list of (address, pulse) pairs.

Gate 8 asks whether the same information can live as a *state* rather than a
ledger.

For each absolute measurement

    y = h^T x

maintain:

    x_hat   current minimum-norm estimate
    Q       orthonormal basis for the span of independent measurement masks

For a new address h, remove the part already represented by Q:

    q = h - Q Q^T h

If q adds an independent direction, update only along q:

    x_hat <- x_hat + (y - h^T x_hat) q / ||q||^2

Because q is orthogonal to all previously retained measurement directions, the
update satisfies the new constraint without changing the old retained ones.
Then q/||q|| becomes another column of Q.

No old scalar pulses are kept.  No old addresses are kept as a chronological
list.  The reference is the *subspace they carved*.

A rank cap R tests how many independent directions are needed.  A wrong-address
attacker feeds the same returned scalar to a different same-scale lens mask.

This is streaming linear algebra / a sufficient-statistic state.  It is not a
claim that real dendrites perform Gram-Schmidt, and it is not Takens' theorem.
It tests the more general idea that the "copy" can be a compact state preserving
intervention geometry rather than a duplicate of the past.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
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
    trial as gate6c_trial,
)
from gate7_forced_history import wrong_same_scale_mask

ROOT = Path(__file__).resolve().parents[1]
RANKS = (1, 2, 3, 4, 8)


@dataclass
class StreamingReference:
    base: np.ndarray
    rank_cap: int
    estimate: np.ndarray
    basis: list[np.ndarray]

    @classmethod
    def create(cls, base: np.ndarray, rank_cap: int) -> "StreamingReference":
        return cls(
            base=base.copy(),
            rank_cap=int(rank_cap),
            estimate=base.copy(),
            basis=[],
        )

    @property
    def rank(self) -> int:
        return len(self.basis)

    def update(self, mask: np.ndarray, absolute: float) -> dict:
        h = np.asarray(mask, dtype=float).reshape(-1)
        q = h.copy()
        for basis_vec in self.basis:
            q -= float(np.dot(basis_vec, q)) * basis_vec

        qnorm2 = float(np.dot(q, q))
        predicted = float(np.dot(h, self.estimate.reshape(-1)))
        residual = float(absolute - predicted)
        added = False

        if qnorm2 > 1e-14 and self.rank < self.rank_cap:
            x = self.estimate.reshape(-1)
            x += (residual / qnorm2) * q
            self.estimate = x.reshape(self.estimate.shape)
            self.basis.append(q / np.sqrt(qnorm2))
            added = True
        elif qnorm2 <= 1e-14:
            # Redundant constraint. With detector noise we deliberately do not
            # rewrite old independent directions; this is the bounded-state
            # analogue of keeping a stable reference rather than chasing noise.
            pass
        else:
            # Rank budget exhausted: the new independent constraint cannot be
            # represented. Keep the protected subspace unchanged.
            pass

        return {
            "innovation": residual,
            "new_direction_norm2": qnorm2,
            "basis_rank": self.rank,
            "added_direction": added,
        }


def streaming_write(
    ref: StreamingReference,
    lens: Lens,
    residual: float,
    lenses: list[Lens],
    scramble_address: bool,
) -> dict:
    # Convert residual pulse back to an absolute measurement before changing
    # the estimate, exactly as Gate 6C HISTORY_REPLAY does.
    absolute = float(np.sum(ref.estimate * lens.mask)) + float(residual)
    mask = wrong_same_scale_mask(lens, lenses) if scramble_address else lens.mask
    return ref.update(mask, absolute)


def search(
    world: np.ndarray,
    base: np.ndarray,
    ref: StreamingReference,
    templates: list[np.ndarray],
    lenses: list[Lens],
    rng: np.random.Generator,
    scramble_address: bool,
) -> dict:
    posterior = np.full(len(templates), 1.0 / len(templates), dtype=float)
    unused = list(range(len(lenses)))
    trace = []

    for _ in range(MAX_SEARCH_PROBES):
        scores = []
        means_by_lens = []
        for lens_index in unused:
            lens = lenses[lens_index]
            means = np.asarray(
                [read(base + template - ref.estimate, lens) for template in templates],
                dtype=float,
            )
            means_by_lens.append(means)
            scores.append(partition_information_bits(posterior, means, PAID_NOISE_SIGMA))

        local_index = int(np.argmax(scores))
        lens_index = int(unused[local_index])
        lens = lenses[lens_index]
        means = means_by_lens[local_index]

        observed = read(world - ref.estimate, lens) + float(
            rng.normal(scale=PAID_NOISE_SIGMA)
        )
        logp = (
            np.log(np.maximum(posterior, 1e-300))
            - 0.5 * ((observed - means) / PAID_NOISE_SIGMA) ** 2
        )
        logp -= float(np.max(logp))
        posterior = np.exp(logp)
        posterior /= float(np.sum(posterior))

        update = streaming_write(ref, lens, observed, lenses, scramble_address)
        trace.append(
            {
                "lens": lens.name,
                "scale": int(lens.scale),
                "posterior_max": float(np.max(posterior)),
                **update,
            }
        )
        unused.remove(lens_index)
        if float(np.max(posterior)) >= POSTERIOR_STOP:
            break

    return {
        "identity": int(np.argmax(posterior)),
        "confidence": float(np.max(posterior)),
        "probes": len(trace),
        "trace": trace,
    }


def trial(identity: int, seed: int, rank_cap: int, scramble_address: bool) -> dict:
    base = np.zeros((N, N), dtype=float)
    templates = make_templates()
    lenses = make_lenses()
    world = base + templates[identity]
    ref = StreamingReference.create(base, rank_cap)

    offset = 1_300_003 if not scramble_address else 1_500_007
    rng = np.random.default_rng(seed * 1009 + identity * 53 + offset + rank_cap * 17)

    remembered_identity = None
    paid_probes = 0
    repeated_paid_steps = 0
    positive_searches = 0
    correct_searches = 0
    negative_home_triggers = 0
    scale_sequences = []

    for step in range(ON_STEPS):
        home_residual = float(np.mean(world - ref.estimate)) + float(
            rng.normal(scale=HOME_NOISE_SIGMA)
        )
        if abs(home_residual) < SURPRISE_THRESHOLD:
            continue
        if step > 0:
            repeated_paid_steps += 1

        if home_residual > 0:
            positive_searches += 1
            result = search(
                world,
                base,
                ref,
                templates,
                lenses,
                rng,
                scramble_address,
            )
            paid_probes += int(result["probes"])
            scale_sequences.append([int(r["scale"]) for r in result["trace"]])
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
                if candidate.scale == 4 and candidate.name == f"s4_x{4 * fine_identity}"
            )
            observed = read(world - ref.estimate, lens) + float(
                rng.normal(scale=PAID_NOISE_SIGMA)
            )
            paid_probes += 1
            if observed < -0.5 * PATCH_AMPLITUDE:
                streaming_write(ref, lens, observed, lenses, scramble_address)

    return {
        "paid_probes": int(paid_probes),
        "repeated_paid_steps": int(repeated_paid_steps),
        "positive_searches": int(positive_searches),
        "correct_searches": int(correct_searches),
        "negative_home_triggers": int(negative_home_triggers),
        "final_mse": float(np.mean((world - ref.estimate) ** 2)),
        "final_home_residual": float(np.mean(world - ref.estimate)),
        "scale_sequences": scale_sequences,
        "final_reference_rank": int(ref.rank),
    }


def run_rank(rank_cap: int, scramble: bool) -> dict:
    rows = [
        trial(identity, seed, rank_cap, scramble)
        for seed in AUDIT_SEEDS
        for identity in range(8)
    ]
    out = aggregate(rows)
    out["rank_cap"] = int(rank_cap)
    out["scramble_address"] = bool(scramble)
    out["final_reference_rank_mean"] = float(
        np.mean([r["final_reference_rank"] for r in rows])
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "gate8",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    local = aggregate([
        gate6c_trial(identity, 0.0, seed, "LOCAL_SAME_FIELD")
        for seed in AUDIT_SEEDS for identity in range(8)
    ])
    full = aggregate([
        gate6c_trial(identity, 0.0, seed, "HISTORY_REPLAY")
        for seed in AUDIT_SEEDS for identity in range(8)
    ])
    correct = [run_rank(r, False) for r in RANKS]
    scrambled = [run_rank(r, True) for r in RANKS]

    summary = {
        "schema": "geometric-neuron/gate8-streaming-reference-subspace-v1",
        "question": (
            "Can the finite addressed reference be stored as an online low-rank subspace state rather than an explicit history list?"
        ),
        "baseline_local_same_field": local,
        "baseline_full_history_replay": full,
        "correct_address": correct,
        "scrambled_address": scrambled,
        "claim_boundary": (
            "The streaming orthogonalization is exact linear algebra, not a biological mechanism. A positive result shows sufficiency of a low-rank intervention subspace state in this toy, not dendritic Gram-Schmidt or Takens implementation."
        ),
    }
    path = args.output_dir / "gate8_summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
