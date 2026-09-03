#!/usr/bin/env python3
"""Gate 6C post-result audit: does same-field WRITE need a timescale gap?

Gate 6 writes a recognized residual into the same spatial representation that
future lenses address, but only after an onset search has finished.  This audit
moves WRITE *between* paid pulses and sweeps its first-order time constant.

A pulse y = h^T(world - x) immediately produces the local correction

    x <- x + alpha * y * h / ||h||^2
    alpha = 1 - exp(- probe_interval / tau_write)

so the addressed spatial state changes before the next pulse.  tau=0 is an
instantaneous write and tau=inf is no write.

Two rules are compared:

LOCAL_SAME_FIELD
    Keeps only the spatial field.  Every pulse backprojects locally into the
    same field, with no separate pulse-history estimator.

HISTORY_REPLAY
    A deliberately boring state-estimator attacker.  It stores every absolute
    pulse equation and recomputes the minimum-norm field consistent with that
    history before applying the same alpha.  This is the "copy/bookkeeping"
    escape hatch the same-field object is trying not to assume.

The world is one of Gate 6's eight 4x4 anomaly locations and remains present
for eight steps.  The HOME pulse and paid-lens noise are unchanged.  Probe
selection uses the high-SNR discrete form of Gate 6B information gain: means
within five detector sigmas are one observable outcome.  With Gate 6's very
large SNR this reproduces the 16 -> 8 -> 4 partition policy while making the
robustness sweep cheap enough for CI.

This is explicitly post-result.  It does not alter Gate 6 or 6B.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

N = 32
PATCH = 4
PATCH_AMPLITUDE = 0.8
HOME_NOISE_SIGMA = 0.001
PAID_NOISE_SIGMA = 0.002
SURPRISE_THRESHOLD = 0.004
POSTERIOR_STOP = 0.995
MAX_SEARCH_PROBES = 8
ON_STEPS = 8
AUDIT_SEEDS = tuple(range(1, 11))
TAUS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, float("inf"))

RULES = ("LOCAL_SAME_FIELD", "HISTORY_REPLAY")


@dataclass(frozen=True)
class Lens:
    name: str
    scale: int
    mask: np.ndarray


def box_mask(y0: int, x0: int, scale: int) -> np.ndarray:
    mask = np.zeros((N, N), dtype=float)
    mask[y0 : y0 + scale, x0 : x0 + scale] = 1.0 / float(scale * scale)
    return mask


def make_lenses() -> list[Lens]:
    lenses: list[Lens] = []
    for x0 in (0, 16):
        lenses.append(Lens(f"s16_x{x0}", 16, box_mask(8, x0, 16)))
    for x0 in (0, 8, 16, 24):
        lenses.append(Lens(f"s8_x{x0}", 8, box_mask(12, x0, 8)))
    for x0 in range(0, N, 4):
        lenses.append(Lens(f"s4_x{x0}", 4, box_mask(12, x0, 4)))
    return lenses


def make_templates() -> list[np.ndarray]:
    templates = []
    for identity in range(8):
        patch = np.zeros((N, N), dtype=float)
        x0 = 4 * identity
        patch[12:16, x0 : x0 + PATCH] = PATCH_AMPLITUDE
        templates.append(patch)
    return templates


def read(field: np.ndarray, lens: Lens) -> float:
    return float(np.sum(field * lens.mask))


def entropy_bits(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def partition_information_bits(
    posterior: np.ndarray,
    means: np.ndarray,
    sigma: float,
) -> float:
    """High-SNR discrete approximation to Gate 6B mutual information."""
    posterior = np.asarray(posterior, dtype=float)
    means = np.asarray(means, dtype=float)
    order = np.argsort(means)

    groups: list[list[int]] = []
    current: list[int] = []
    previous: float | None = None
    for index in order:
        value = float(means[index])
        if previous is None or abs(value - previous) <= 5.0 * sigma:
            current.append(int(index))
        else:
            groups.append(current)
            current = [int(index)]
        previous = value
    if current:
        groups.append(current)

    outcome = np.asarray(
        [sum(float(posterior[i]) for i in group) for group in groups],
        dtype=float,
    )
    return entropy_bits(outcome)


def alpha_from_tau(tau: float) -> float:
    if tau == 0.0:
        return 1.0
    if math.isinf(tau):
        return 0.0
    return float(1.0 - math.exp(-1.0 / tau))


def local_write(
    memory: np.ndarray,
    lens: Lens,
    residual: float,
    alpha: float,
) -> None:
    norm2 = float(np.sum(lens.mask * lens.mask))
    memory += alpha * float(residual) * lens.mask / max(norm2, 1e-30)


def solve_history(
    base: np.ndarray,
    masks: list[np.ndarray],
    values: list[float],
) -> np.ndarray:
    if not masks:
        return base.copy()
    H = np.stack([mask.reshape(-1) for mask in masks], axis=0)
    z = np.asarray(values, dtype=float) - H @ base.reshape(-1)
    gram = H @ H.T
    coeff = np.linalg.pinv(gram, rcond=1e-10) @ z
    return base + (H.T @ coeff).reshape(base.shape)


def apply_write(
    rule: str,
    base: np.ndarray,
    memory: np.ndarray,
    lens: Lens,
    residual: float,
    alpha: float,
    history_masks: list[np.ndarray],
    history_values: list[float],
) -> None:
    if rule == "LOCAL_SAME_FIELD":
        local_write(memory, lens, residual, alpha)
        return

    # Convert residual pulse back to an absolute world measurement before
    # changing memory. HISTORY_REPLAY deliberately keeps this extra record.
    absolute = float(np.sum(memory * lens.mask)) + float(residual)
    history_masks.append(lens.mask.copy())
    history_values.append(absolute)
    target = solve_history(base, history_masks, history_values)
    memory[:] = (1.0 - alpha) * memory + alpha * target


def search(
    world: np.ndarray,
    base: np.ndarray,
    memory: np.ndarray,
    templates: list[np.ndarray],
    lenses: list[Lens],
    rng: np.random.Generator,
    alpha: float,
    rule: str,
    history_masks: list[np.ndarray],
    history_values: list[float],
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
                partition_information_bits(
                    posterior,
                    means,
                    PAID_NOISE_SIGMA,
                )
            )

        local_index = int(np.argmax(scores))
        lens_index = int(unused[local_index])
        lens = lenses[lens_index]
        means = means_by_lens[local_index]

        observed = (
            read(world - memory, lens)
            + float(rng.normal(scale=PAID_NOISE_SIGMA))
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

        # The point of Gate 6C: WRITE happens before the next READ.
        apply_write(
            rule,
            base,
            memory,
            lens,
            observed,
            alpha,
            history_masks,
            history_values,
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


def trial(identity: int, tau: float, seed: int, rule: str) -> dict:
    base = np.zeros((N, N), dtype=float)
    templates = make_templates()
    lenses = make_lenses()
    world = base + templates[identity]
    memory = base.copy()
    alpha = alpha_from_tau(tau)

    offset = 0 if rule == "LOCAL_SAME_FIELD" else 100003
    rng = np.random.default_rng(seed * 1009 + identity * 53 + offset)

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
        home_residual = (
            float(np.mean(world - memory))
            + float(rng.normal(scale=HOME_NOISE_SIGMA))
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
                memory,
                templates,
                lenses,
                rng,
                alpha,
                rule,
                history_masks,
                history_values,
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
            # Fast local writes can make HOME overshoot negative even though
            # the anomaly is still physically present. Pay one exact-patch
            # verification, matching Gate 6's disappearance discipline.
            negative_home_triggers += 1
            fine_identity = 0 if remembered_identity is None else remembered_identity
            lens = next(
                candidate
                for candidate in lenses
                if candidate.scale == 4
                and candidate.name == f"s4_x{4 * fine_identity}"
            )
            observed = (
                read(world - memory, lens)
                + float(rng.normal(scale=PAID_NOISE_SIGMA))
            )
            paid_probes += 1

            # Only an actually negative exact-patch residual is disappearance
            # evidence. A global HOME overshoot alone must not erase the patch.
            if observed < -0.5 * PATCH_AMPLITUDE:
                apply_write(
                    rule,
                    base,
                    memory,
                    lens,
                    observed,
                    alpha,
                    history_masks,
                    history_values,
                )

    return {
        "paid_probes": int(paid_probes),
        "repeated_paid_steps": int(repeated_paid_steps),
        "positive_searches": int(positive_searches),
        "correct_searches": int(correct_searches),
        "negative_home_triggers": int(negative_home_triggers),
        "final_mse": float(np.mean((world - memory) ** 2)),
        "final_home_residual": float(np.mean(world - memory)),
        "alpha": float(alpha),
        "scale_sequences": scale_sequences,
    }


def aggregate(rows: list[dict]) -> dict:
    attempts = sum(int(row["positive_searches"]) for row in rows)
    correct = sum(int(row["correct_searches"]) for row in rows)
    sequence_counts: dict[str, int] = {}
    for row in rows:
        for sequence in row["scale_sequences"]:
            key = "->".join(str(int(scale)) for scale in sequence)
            sequence_counts[key] = sequence_counts.get(key, 0) + 1

    return {
        "trials": int(len(rows)),
        "paid_probes_mean": float(np.mean([row["paid_probes"] for row in rows])),
        "paid_probes_min": int(np.min([row["paid_probes"] for row in rows])),
        "paid_probes_max": int(np.max([row["paid_probes"] for row in rows])),
        "repeated_paid_steps_mean": float(
            np.mean([row["repeated_paid_steps"] for row in rows])
        ),
        "localization_accuracy": float(correct / max(attempts, 1)),
        "negative_home_triggers_mean": float(
            np.mean([row["negative_home_triggers"] for row in rows])
        ),
        "final_mse_mean": float(np.mean([row["final_mse"] for row in rows])),
        "final_home_residual_mean": float(
            np.mean([row["final_home_residual"] for row in rows])
        ),
        "scale_sequences": dict(
            sorted(sequence_counts.items(), key=lambda item: (-item[1], item[0]))
        ),
    }


def tau_label(tau: float) -> str:
    return "inf" if math.isinf(tau) else f"{tau:g}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "gate6c",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tables: dict[str, dict[str, dict]] = {}
    for rule in RULES:
        table: dict[str, dict] = {}
        for tau in TAUS:
            rows = [
                trial(identity, tau, seed, rule)
                for seed in AUDIT_SEEDS
                for identity in range(8)
            ]
            table[tau_label(tau)] = aggregate(rows)
        tables[rule] = table

    local = tables["LOCAL_SAME_FIELD"]
    history = tables["HISTORY_REPLAY"]
    best_local_tau = min(
        TAUS,
        key=lambda tau: local[tau_label(tau)]["paid_probes_mean"],
    )
    best_history_tau = min(
        TAUS,
        key=lambda tau: history[tau_label(tau)]["paid_probes_mean"],
    )
    best_local_cost = local[tau_label(best_local_tau)]["paid_probes_mean"]

    summary = {
        "best_local_tau_over_probe_interval": tau_label(best_local_tau),
        "best_local_paid_probes": float(best_local_cost),
        "instant_local_paid_probes": float(local["0"]["paid_probes_mean"]),
        "no_write_local_paid_probes": float(local["inf"]["paid_probes_mean"]),
        "instant_over_best_local_cost_ratio": float(
            local["0"]["paid_probes_mean"] / best_local_cost
        ),
        "no_write_over_best_local_cost_ratio": float(
            local["inf"]["paid_probes_mean"] / best_local_cost
        ),
        "best_history_tau_over_probe_interval": tau_label(best_history_tau),
        "instant_history_paid_probes": float(history["0"]["paid_probes_mean"]),
        "minimum_localization_accuracy_local": float(
            min(row["localization_accuracy"] for row in local.values())
        ),
        "minimum_localization_accuracy_history": float(
            min(row["localization_accuracy"] for row in history.values())
        ),
    }

    tradeoff = (
        summary["instant_over_best_local_cost_ratio"] > 1.15
        and summary["no_write_over_best_local_cost_ratio"] > 2.0
        and summary["instant_history_paid_probes"]
        < summary["instant_local_paid_probes"]
    )
    classification = (
        "INTERMEDIATE_LOCAL_WRITE_TIMESCALE_MINIMIZES_PROBES_"
        "BUT_HISTORY_ATTACKER_REMOVES_FAST_WRITE_PENALTY"
        if tradeoff
        else "WRITE_TIMESCALE_TRADEOFF_NOT_ESTABLISHED"
    )
    copy_worry = (
        "IDENTITY_EVIDENCE_NOT_CORRUPTED_IN_THIS_HIGH_SNR_TOY"
        if summary["minimum_localization_accuracy_local"] >= 0.99
        else "FAST_WRITE_DEGRADES_IDENTITY_EVIDENCE"
    )

    result = {
        "audit": "gate6c_write_timescale",
        "post_result": True,
        "classification": classification,
        "copy_worry": copy_worry,
        "probe_interval_units": 1.0,
        "write_rule": (
            "x <- x + alpha * residual * h / ||h||^2; "
            "alpha = 1-exp(-1/tau)"
        ),
        "tau_over_probe_interval": [tau_label(tau) for tau in TAUS],
        "rules": tables,
        "summary": summary,
        "interpretation_boundary": (
            "The local same-field writer has a real control-timescale tradeoff, "
            "but fast WRITE did not destroy identity evidence here. Its cost "
            "comes from coarse/fine backprojections overshooting HOME. The "
            "history-replay state-estimator attacker removes that fast-write "
            "penalty by keeping extra measurement bookkeeping."
        ),
    }

    output_path = args.output_dir / "gate6c.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print("GeometricNeuronV24 Gate 6C - same-field WRITE timescale")
    print()
    for rule in RULES:
        print(rule)
        for tau in TAUS:
            row = tables[rule][tau_label(tau)]
            print(
                f"  tau={tau_label(tau):>4}  "
                f"paid={row['paid_probes_mean']:6.3f}  "
                f"repeat={row['repeated_paid_steps_mean']:5.3f}  "
                f"negative_HOME={row['negative_home_triggers_mean']:5.3f}  "
                f"identity={row['localization_accuracy']:.3f}"
            )
        print()
    print(json.dumps(summary, indent=2))
    print(classification)
    print(copy_worry)

    raise SystemExit(0 if tradeoff else 1)


if __name__ == "__main__":
    main()
