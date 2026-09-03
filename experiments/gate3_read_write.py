#!/usr/bin/env python3
"""Gate 3: READ+WRITE turns hidden operator differences into addressed evidence.

3A keeps the operator fixed but starts from an exactly uninformative state. Four
periodic transport laws preserve both zero and uniform fields, so read-only and
an equal-energy uniform write are exact nulls. A localized state write excites
the hidden operator; addressed scalar reads can then identify which transport
law is present.

3B separates state writes from operator writes. One of four local regions gets
a persistent retention change. The fast field and the write identity are then
erased. A public diagnostic field is loaded and only scalar addressed reads are
allowed. The persistent operator modification remains identifiable. The sensor
noise is deliberately heteroscedastic across lens scales, making this also a
control against naive raw-variance probe selection.

This is a computational observability experiment. It is not evidence that a
biological neuron has an address bus, performs physical backpropagation, or
implements this exact growth law.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class TransportLaw:
    name: str
    dy: int
    dx: int


def area_overlap_1d(n: int, scale: int) -> np.ndarray:
    source_left = np.arange(n, dtype=np.float64) / n
    source_right = np.arange(1, n + 1, dtype=np.float64) / n
    target_left = np.arange(scale, dtype=np.float64) / scale
    target_right = np.arange(1, scale + 1, dtype=np.float64) / scale
    overlap = np.maximum(
        0.0,
        np.minimum(target_right[:, None], source_right[None, :])
        - np.maximum(target_left[:, None], source_left[None, :]),
    )
    return scale * overlap


def lens_matrix(n: int, scale: int) -> np.ndarray:
    p = area_overlap_1d(n, scale)
    return np.einsum("iy,jx->ijyx", p, p, optimize=True).reshape(scale * scale, n * n)


def candidate_lenses(side: int, scales: tuple[int, ...]) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    matrices = []
    addresses: list[tuple[int, int, int]] = []
    for scale in scales:
        matrices.append(lens_matrix(side, scale))
        addresses.extend((scale, row, col) for row in range(scale) for col in range(scale))
    return np.vstack(matrices), addresses


def normalize_log_weights(log_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    maximum = np.max(log_weights, axis=1, keepdims=True)
    weights = np.exp(log_weights - maximum)
    weights /= np.sum(weights, axis=1, keepdims=True)
    return np.log(np.maximum(weights, 1e-300)), weights


def transport_step(field: np.ndarray, law: TransportLaw, diffusion: float = 0.08) -> np.ndarray:
    moved = np.roll(field, shift=(law.dy, law.dx), axis=(0, 1))
    laplacian = (
        np.roll(moved, 1, axis=0)
        + np.roll(moved, -1, axis=0)
        + np.roll(moved, 1, axis=1)
        + np.roll(moved, -1, axis=1)
        - 4.0 * moved
    )
    return moved + diffusion * laplacian


def transport_predictions(initial: np.ndarray, laws: list[TransportLaw], lens: np.ndarray, steps: int) -> np.ndarray:
    trajectories = []
    for law in laws:
        field = initial.copy()
        history = []
        for _ in range(steps):
            field = transport_step(field, law)
            history.append(field.reshape(-1).copy())
        trajectories.append(history)
    trajectories = np.asarray(trajectories, dtype=np.float64)
    return np.einsum("htp,ap->tha", trajectories, lens, optimize=True)


def max_hypothesis_spread(predictions: np.ndarray) -> float:
    return float(np.max(np.ptp(predictions, axis=1)))


def run_finite_hypothesis_policy(
    predictions: np.ndarray,
    truth: np.ndarray,
    rng: np.random.Generator,
    noise_sigma: np.ndarray,
    policy: str,
) -> dict:
    steps, hypotheses, candidates = predictions.shape
    trials = truth.size
    log_weights = np.full((trials, hypotheses), -np.log(hypotheses), dtype=np.float64)
    accuracy_curve: list[float] = []
    entropy_curve: list[float] = []
    choices_all: list[int] = []

    for time in range(steps):
        log_weights, weights = normalize_log_weights(log_weights)
        choices = np.empty(trials, dtype=int)
        table = predictions[time]
        for trial in range(trials):
            if policy == "random":
                choices[trial] = int(rng.integers(0, candidates))
                continue
            mean = np.sum(weights[trial, :, None] * table, axis=0)
            variance = np.sum(weights[trial, :, None] * (table - mean) ** 2, axis=0)
            if policy == "raw_variance":
                score = variance
            elif policy in {"active", "noise_aware"}:
                score = variance / np.maximum(noise_sigma**2, 1e-18)
            else:
                raise ValueError(f"unknown policy: {policy}")
            choices[trial] = int(np.argmax(score))

        sigma = noise_sigma[choices]
        observed = predictions[time, truth, choices] + rng.normal(scale=sigma, size=trials)
        predicted = np.stack(
            [predictions[time, :, choices[trial]] for trial in range(trials)], axis=0
        )
        residual = observed[:, None] - predicted
        log_weights += -0.5 * (residual / sigma[:, None]) ** 2
        _, weights = normalize_log_weights(log_weights)
        estimates = np.argmax(weights, axis=1)
        accuracy_curve.append(float(np.mean(estimates == truth)))
        entropy = -np.sum(weights * np.log2(np.maximum(weights, 1e-300)), axis=1)
        entropy_curve.append(float(np.mean(entropy)))
        choices_all.extend(int(item) for item in choices)

    _, weights = normalize_log_weights(log_weights)
    estimates = np.argmax(weights, axis=1)
    return {
        "final_accuracy": float(np.mean(estimates == truth)),
        "final_entropy_bits": float(entropy_curve[-1]),
        "accuracy_curve": accuracy_curve,
        "entropy_curve": entropy_curve,
        "unique_addresses_used": int(len(set(choices_all))),
    }


def gate3a(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    side = 16
    steps = 4
    trials = 256
    laws = [
        TransportLaw("east", 0, 1),
        TransportLaw("south", 1, 0),
        TransportLaw("west", 0, -1),
        TransportLaw("north", -1, 0),
    ]
    lens, addresses = candidate_lenses(side, (4, 8, 16))

    zero = np.zeros((side, side), dtype=np.float64)
    uniform = np.ones((side, side), dtype=np.float64) / side
    localized = np.zeros((side, side), dtype=np.float64)
    localized[side // 2, side // 2] = 1.0

    zero_predictions = transport_predictions(zero, laws, lens, steps)
    uniform_predictions = transport_predictions(uniform, laws, lens, steps)
    localized_predictions = transport_predictions(localized, laws, lens, steps)

    truth = rng.integers(0, len(laws), size=trials)
    sigma = np.full(len(addresses), 0.01, dtype=np.float64)
    active = run_finite_hypothesis_policy(
        localized_predictions, truth, np.random.default_rng(seed + 101), sigma, "active"
    )
    random_result = run_finite_hypothesis_policy(
        localized_predictions, truth, np.random.default_rng(seed + 202), sigma, "random"
    )

    zero_spread = max_hypothesis_spread(zero_predictions)
    uniform_spread = max_hypothesis_spread(uniform_predictions)
    local_spread = max_hypothesis_spread(localized_predictions)
    requirements = {
        "read_only_is_exact_null": zero_spread < 1e-13,
        "equal_energy_uniform_write_is_exact_null": uniform_spread < 1e-13,
        "localized_write_excites_hidden_operator": local_spread > 0.25,
        "active_identification_ge_0_99": active["final_accuracy"] >= 0.99,
        "active_beats_random_by_0_35": active["final_accuracy"] - random_result["final_accuracy"] >= 0.35,
    }
    passed = all(requirements.values())
    return {
        "name": "gate3a_state_write_reveals_hidden_transport",
        "passed": passed,
        "classification": "LOCALIZED_STATE_WRITE_BREAKS_EXACT_READ_NULL" if passed else "GATE3A_FAILED",
        "seed": seed,
        "side": side,
        "steps": steps,
        "trials": trials,
        "laws": [asdict(law) for law in laws],
        "write_l2_energy": {
            "uniform": float(np.linalg.norm(uniform)),
            "localized": float(np.linalg.norm(localized)),
        },
        "max_hypothesis_spread": {
            "read_only_zero": zero_spread,
            "uniform_write": uniform_spread,
            "localized_write": local_spread,
        },
        "active": active,
        "random": random_result,
        "locked_requirements": requirements,
    }


def structural_response(
    side: int,
    patch: tuple[int, int],
    steps: int,
    diffusion: float = 0.10,
    baseline_retention: float = 0.65,
    structural_boost: float = 0.40,
) -> np.ndarray:
    retention = np.full((side, side), baseline_retention, dtype=np.float64)
    row, col = patch
    retention[row : row + 4, col : col + 4] += structural_boost

    # The original fast field has been erased. This public diagnostic load is
    # identical for every hidden structural-write hypothesis.
    field = np.ones((side, side), dtype=np.float64) / side
    history = []
    for _ in range(steps):
        laplacian = (
            np.roll(field, 1, axis=0)
            + np.roll(field, -1, axis=0)
            + np.roll(field, 1, axis=1)
            + np.roll(field, -1, axis=1)
            - 4.0 * field
        )
        field = (field + diffusion * laplacian) * retention
        history.append(field.reshape(-1).copy())
    return np.asarray(history)


def gate3b(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    side = 16
    steps = 12
    trials = 256
    patches = [(2, 2), (2, 10), (10, 2), (10, 10)]
    scales = (2, 4, 8, 16)
    lens, addresses = candidate_lenses(side, scales)

    trajectories = np.asarray(
        [structural_response(side, patch, steps) for patch in patches], dtype=np.float64
    )
    predictions = np.einsum("htp,ap->tha", trajectories, lens, optimize=True)

    # Deliberate heteroscedastic detector: fine lenses have higher pulse noise.
    # The observer knows this calibration, so raw variance is not optimal.
    noise_sigma = np.asarray(
        [0.001 + 0.30 * (scale / side) ** 3 for scale, _, _ in addresses],
        dtype=np.float64,
    )
    truth = rng.integers(0, len(patches), size=trials)
    noise_aware = run_finite_hypothesis_policy(
        predictions, truth, np.random.default_rng(seed + 303), noise_sigma, "noise_aware"
    )
    raw_variance = run_finite_hypothesis_policy(
        predictions, truth, np.random.default_rng(seed + 404), noise_sigma, "raw_variance"
    )
    random_result = run_finite_hypothesis_policy(
        predictions, truth, np.random.default_rng(seed + 505), noise_sigma, "random"
    )

    baseline = structural_response(side, patches[0], steps, structural_boost=0.0)
    baseline_predictions = np.einsum("tp,ap->ta", baseline, lens, optimize=True)
    no_write_spread = float(
        np.max(
            np.ptp(
                np.repeat(baseline_predictions[:, None, :], len(patches), axis=1),
                axis=1,
            )
        )
    )
    structural_spread = max_hypothesis_spread(predictions)

    requirements = {
        "no_structural_write_is_exact_null": no_write_spread < 1e-13,
        "persistent_operator_write_survives_reset": structural_spread > 0.005,
        "noise_aware_identification_ge_0_99": noise_aware["final_accuracy"] >= 0.99,
        "noise_aware_beats_random_by_0_45": noise_aware["final_accuracy"] - random_result["final_accuracy"] >= 0.45,
        "raw_variance_not_close_to_noise_aware": raw_variance["final_accuracy"] <= 0.75,
    }
    passed = all(requirements.values())
    return {
        "name": "gate3b_persistent_operator_write_after_fast_state_erasure",
        "passed": passed,
        "classification": (
            "PERSISTENT_OPERATOR_WRITE_REMAINS_ADDRESSABLY_OBSERVABLE_AFTER_RESET"
            if passed
            else "GATE3B_FAILED"
        ),
        "seed": seed,
        "side": side,
        "steps": steps,
        "trials": trials,
        "hidden_write_regions": [list(item) for item in patches],
        "fast_state_erased_before_diagnostic": True,
        "write_identity_available_to_observer": False,
        "max_hypothesis_spread": {
            "no_operator_write": no_write_spread,
            "persistent_operator_write": structural_spread,
        },
        "noise_sigma_by_scale": {
            str(scale): float(0.001 + 0.30 * (scale / side) ** 3) for scale in scales
        },
        "noise_aware": noise_aware,
        "raw_variance": raw_variance,
        "random": random_result,
        "locked_requirements": requirements,
    }


def run(seed: int, output_dir: Path) -> dict:
    result_a = gate3a(seed)
    result_b = gate3b(seed)
    passed = bool(result_a["passed"] and result_b["passed"])
    result = {
        "gate": 3,
        "name": "read_write_observability",
        "seed": seed,
        "passed": passed,
        "classification": (
            "READ_WRITE_SEPARATES_STATE_EXCITATION_FROM_PERSISTENT_OPERATOR_CHANGE"
            if passed
            else "GATE3_FAILED"
        ),
        "gate3a": result_a,
        "gate3b": result_b,
        "scope": (
            "Gate 3A writes fast state but leaves the operator fixed. Gate 3B "
            "writes the operator, then erases the fast state and write identity "
            "before a public diagnostic load. Both are finite controlled law families."
        ),
        "biological_boundary": (
            "A diagnostic state write is analogous only at the level of experimental "
            "intervention; a persistent operator write is analogous only at the level "
            "of lasting local plasticity. No backpropagating spike is identified with "
            "an adjoint, and no neuron claim is earned by this image-field laboratory."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "gate3.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def print_receipt(result: dict) -> None:
    a = result["gate3a"]
    b = result["gate3b"]
    print("GeometricNeuronV24 Gate 3 - READ + WRITE")
    print()
    print("Gate 3A - state write / hidden transport")
    print(f"  read-only max law spread:           {a['max_hypothesis_spread']['read_only_zero']:.3e}")
    print(f"  uniform-write max law spread:       {a['max_hypothesis_spread']['uniform_write']:.3e}")
    print(f"  localized-write max law spread:     {a['max_hypothesis_spread']['localized_write']:.6f}")
    print(f"  active / random exact accuracy:     {a['active']['final_accuracy']:.3f} / {a['random']['final_accuracy']:.3f}")
    print()
    print("Gate 3B - persistent operator write")
    print(f"  no-write max law spread:            {b['max_hypothesis_spread']['no_operator_write']:.3e}")
    print(f"  structural-write max law spread:    {b['max_hypothesis_spread']['persistent_operator_write']:.6f}")
    print(f"  noise-aware exact accuracy:         {b['noise_aware']['final_accuracy']:.3f}")
    print(f"  raw-variance / random accuracy:     {b['raw_variance']['final_accuracy']:.3f} / {b['random']['final_accuracy']:.3f}")
    print()
    print(result["classification"])
    print(
        "A localized state write can reveal a hidden operator without changing it; "
        "a persistent operator write remains observable after fast-state erasure. "
        "This is an observability result, not a biological-neuron or physical-adjoint claim."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=24031976)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "gate3")
    args = parser.parse_args()
    result = run(args.seed, args.output_dir)
    print_receipt(result)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
