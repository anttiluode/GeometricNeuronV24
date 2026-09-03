#!/usr/bin/env python3
"""Gate 2: infer a hidden spatial law from pulse + address history.

A known Gaussian seed evolves under one of 64 hidden transport/growth laws.  At
each time step the observer receives exactly one noisy scalar box average.  The
observer may keep one sensor fixed, choose random addresses, or choose the next
scale/address that maximizes disagreement among its remaining hypotheses.

This is intentionally a finite hypothesis laboratory, not a claim that an
arbitrary PDE can be recovered from one channel.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnv24.lens import lens_matrix  # noqa: E402


@dataclass(frozen=True)
class Law:
    direction: int
    speed: float
    diffusion: float
    decay: float


@dataclass(frozen=True)
class ProbeAddress:
    scale: int
    row: int
    col: int


def law_table() -> list[Law]:
    return [
        Law(direction, speed, diffusion, decay)
        for direction in range(8)
        for speed in (0.014, 0.024)
        for diffusion in (0.00018, 0.00062)
        for decay in (0.004, 0.032)
    ]


def candidate_lenses(side: int, scales: list[int]) -> tuple[np.ndarray, list[ProbeAddress]]:
    matrices = []
    addresses: list[ProbeAddress] = []
    for scale in scales:
        matrices.append(lens_matrix(side, scale))
        addresses.extend(
            ProbeAddress(scale, row, col)
            for row in range(scale)
            for col in range(scale)
        )
    return np.vstack(matrices), addresses


def render_laws(side: int, laws: list[Law], steps: int) -> np.ndarray:
    """Return fields with shape ``(steps, hypotheses, pixels)``."""

    axis = (np.arange(side, dtype=np.float64) + 0.5) / side
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    fields = np.empty((steps, len(laws), side * side), dtype=np.float64)
    sigma0_sq = 0.055**2
    for time in range(1, steps + 1):
        for index, law in enumerate(laws):
            angle = 2.0 * np.pi * law.direction / 8.0
            cx = 0.5 + law.speed * time * np.cos(angle)
            cy = 0.5 + law.speed * time * np.sin(angle)
            sigma_sq = sigma0_sq + 2.0 * law.diffusion * time
            amplitude = np.exp(-law.decay * time)
            field = amplitude * np.exp(
                -0.5 * ((xx - cx) ** 2 + (yy - cy) ** 2) / sigma_sq
            )
            fields[time - 1, index] = field.reshape(-1)
    return fields


def normalize_log_weights(log_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    maximum = np.max(log_weights, axis=1, keepdims=True)
    weights = np.exp(log_weights - maximum)
    weights /= np.sum(weights, axis=1, keepdims=True)
    return np.log(np.maximum(weights, 1e-300)), weights


def active_choices(predictions: np.ndarray, weights: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """Weighted predictive variance for every trial and address."""

    permitted = predictions[:, allowed]
    means = weights @ permitted
    second = weights @ (permitted**2)
    variance = np.maximum(second - means**2, 0.0)
    local = np.argmax(variance, axis=1)
    return allowed[local]


def run_policy(
    policy: str,
    rng: np.random.Generator,
    truth: np.ndarray,
    predictions: np.ndarray,
    addresses: list[ProbeAddress],
    noise_sigma: float,
) -> dict:
    steps, hypotheses, candidates = predictions.shape
    trials = len(truth)
    log_weights = np.full((trials, hypotheses), -np.log(hypotheses), dtype=np.float64)
    accuracy_curve = []
    direction_curve = []
    entropy_curve = []
    chosen_scales: Counter[int] = Counter()
    fixed_index = addresses.index(ProbeAddress(8, 3, 3))
    fixed_scale = np.asarray(
        [i for i, address in enumerate(addresses) if address.scale == 8], dtype=int
    )
    all_indices = np.arange(candidates, dtype=int)
    trial_indices = np.arange(trials)

    for time in range(steps):
        log_weights, weights = normalize_log_weights(log_weights)
        if policy == "fixed":
            choices = np.full(trials, fixed_index, dtype=int)
        elif policy == "random":
            choices = rng.integers(0, candidates, size=trials)
        elif policy == "active_scale8":
            choices = active_choices(predictions[time], weights, fixed_scale)
        elif policy == "active_multiscale":
            choices = active_choices(predictions[time], weights, all_indices)
        else:
            raise ValueError(f"unknown policy: {policy}")

        for choice in choices:
            chosen_scales[addresses[int(choice)].scale] += 1
        expected = predictions[time, truth, choices]
        observed = expected + rng.normal(scale=noise_sigma, size=trials)
        predicted_for_choice = predictions[time][:, choices].T
        residual = observed[:, None] - predicted_for_choice
        log_weights += -0.5 * (residual / noise_sigma) ** 2

        _, weights = normalize_log_weights(log_weights)
        estimates = np.argmax(weights, axis=1)
        accuracy_curve.append(float(np.mean(estimates == truth)))
        direction_curve.append(float(np.mean((estimates // 8) == (truth // 8))))
        entropy = -np.sum(weights * np.log2(np.maximum(weights, 1e-300)), axis=1)
        entropy_curve.append(float(np.mean(entropy)))

    _, weights = normalize_log_weights(log_weights)
    estimates = np.argmax(weights, axis=1)
    # Law ordering: direction, speed, diffusion, decay, with the last three
    # binary dimensions packed into the low three bits.
    truth_direction = truth // 8
    estimate_direction = estimates // 8
    truth_remainder = truth % 8
    estimate_remainder = estimates % 8
    return {
        "exact_accuracy_curve": accuracy_curve,
        "direction_accuracy_curve": direction_curve,
        "entropy_bits_curve": entropy_curve,
        "final_exact_accuracy": float(np.mean(estimates == truth)),
        "final_direction_accuracy": float(np.mean(estimate_direction == truth_direction)),
        "final_speed_accuracy": float(np.mean((estimate_remainder // 4) == (truth_remainder // 4))),
        "final_diffusion_accuracy": float(
            np.mean(((estimate_remainder // 2) % 2) == ((truth_remainder // 2) % 2))
        ),
        "final_decay_accuracy": float(np.mean((estimate_remainder % 2) == (truth_remainder % 2))),
        "final_entropy_bits": entropy_curve[-1],
        "chosen_scale_counts": {str(k): int(v) for k, v in sorted(chosen_scales.items())},
    }


def run(seed: int, output_dir: Path, make_plots: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    side = 16
    steps = 12
    trials = 192
    noise_sigma = 0.012
    scales = [2, 3, 4, 5, 7, 8, 11, 16]
    laws = law_table()
    lens, addresses = candidate_lenses(side, scales)
    fields = render_laws(side, laws, steps)
    predictions = np.einsum("thp,cp->thc", fields, lens, optimize=True)
    truth = rng.integers(0, len(laws), size=trials)

    policies = ["fixed", "random", "active_scale8", "active_multiscale"]
    policy_results = {
        policy: run_policy(
            policy,
            np.random.default_rng(seed + 100 + index),
            truth,
            predictions,
            addresses,
            noise_sigma,
        )
        for index, policy in enumerate(policies)
    }

    active = policy_results["active_multiscale"]
    random_result = policy_results["random"]
    fixed = policy_results["fixed"]
    active8 = policy_results["active_scale8"]
    requirements = {
        "active_exact_accuracy_ge_0_90": active["final_exact_accuracy"] >= 0.90,
        "active_beats_random_by_0_20": (
            active["final_exact_accuracy"] - random_result["final_exact_accuracy"] >= 0.20
        ),
        "active_beats_fixed_by_0_40": (
            active["final_exact_accuracy"] - fixed["final_exact_accuracy"] >= 0.40
        ),
        "active_entropy_below_0_5_bits": active["final_entropy_bits"] <= 0.5,
        "multiscale_not_worse_than_active_scale8": (
            active["final_exact_accuracy"] >= active8["final_exact_accuracy"] - 0.02
        ),
    }
    passed = all(requirements.values())
    result = {
        "gate": 2,
        "name": "hidden_rule_from_addressed_pulses",
        "classification": (
            "ADDRESS_AND_PULSE_HISTORY_IDENTIFY_CONTROLLED_HIDDEN_LAW"
            if passed
            else "HIDDEN_RULE_GATE_FAILED"
        ),
        "passed": passed,
        "seed": seed,
        "image_side": side,
        "hypotheses": len(laws),
        "laws": [asdict(law) for law in laws],
        "trials": trials,
        "steps": steps,
        "noise_sigma": noise_sigma,
        "candidate_scales": scales,
        "candidate_addresses": len(addresses),
        "policies": policy_results,
        "locked_requirements": requirements,
        "scope": (
            "The initial field and 64-member law family are known. Only the law "
            "identity is hidden. One noisy scalar is observed per time step."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "gate2.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if make_plots:
        plot_results(result, fields, truth, output_dir)
    return result


def plot_results(result: dict, fields: np.ndarray, truth: np.ndarray, output_dir: Path) -> None:
    colors = {
        "fixed": "#6c757d",
        "random": "#4ea8de",
        "active_scale8": "#f4a261",
        "active_multiscale": "#52b788",
    }
    steps = np.arange(1, result["steps"] + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    for policy, values in result["policies"].items():
        axes[0].plot(
            steps,
            values["exact_accuracy_curve"],
            marker="o",
            markersize=3,
            color=colors[policy],
            label=policy.replace("_", " "),
        )
        axes[1].plot(
            steps,
            values["entropy_bits_curve"],
            marker="o",
            markersize=3,
            color=colors[policy],
            label=policy.replace("_", " "),
        )
    axes[0].axhline(1 / result["hypotheses"], color="black", linestyle="--", linewidth=1)
    axes[0].set(xlabel="scalar pulses", ylabel="exact hidden-law accuracy", ylim=(-0.02, 1.03))
    axes[1].set(xlabel="scalar pulses", ylabel="posterior entropy (bits)")
    axes[0].set_title("Which law moves the hidden field?")
    axes[1].set_title("The next address collapses hypotheses")
    for axis in axes:
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(output_dir / "gate2_rule_identification.png", dpi=180)
    plt.close(fig)

    side = result["image_side"]
    law_index = int(truth[0])
    times = [0, 3, 7, 11]
    fig, axes = plt.subplots(1, len(times), figsize=(9, 2.5), constrained_layout=True)
    for axis, time in zip(axes, times):
        axis.imshow(fields[time, law_index].reshape(side, side), cmap="magma", vmin=0)
        axis.set_title(f"t = {time + 1}")
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle("One hidden member of the controlled 64-law family")
    fig.savefig(output_dir / "gate2_hidden_dynamics.png", dpi=180)
    plt.close(fig)


def print_receipt(result: dict) -> None:
    print("GeometricNeuronV24 Gate 2 - rule Sherlock")
    print()
    print(f"hidden law hypotheses:                {result['hypotheses']}")
    print(f"trials / scalar pulses:               {result['trials']} / {result['steps']}")
    for policy, values in result["policies"].items():
        print(
            f"  {policy:20s} exact {values['final_exact_accuracy']:.3f}  "
            f"direction {values['final_direction_accuracy']:.3f}  "
            f"entropy {values['final_entropy_bits']:.3f} bits"
        )
    print(
        "active scale use:                    "
        f"{result['policies']['active_multiscale']['chosen_scale_counts']}"
    )
    print()
    print(result["classification"])
    print(
        "This earns active identification only for a known finite law family. "
        "It does not reconstruct an arbitrary image, discover an unconstrained PDE, "
        "or implement a physical adjoint."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=24031976)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "gate2")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    result = run(args.seed, args.output_dir, make_plots=not args.no_plots)
    print_receipt(result)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
