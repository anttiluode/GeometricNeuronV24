#!/usr/bin/env python3
"""Gate 0: establish exactly what the multiscale scalar lens can observe.

This gate is deliberately static.  It does not contain a neuron, growth,
learning, or an adaptive controller.  It first earns the measurement algebra
that later experiments may use.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import qr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnv24.lens import (  # noqa: E402
    adjoint_gradient,
    lens_matrix,
    numerical_rank,
    orthonormal_row_basis,
    pair_rank_prediction,
    stack_lenses,
)


@dataclass
class PairResult:
    scales: tuple[int, int]
    rows: int
    observed_rank: int
    predicted_rank: int
    gain_over_finer_alone: int


def unit_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-15)


def pivoted_row_design(candidates: np.ndarray, count: int) -> np.ndarray:
    """Geometry-only D-optimal-ish row ordering via pivoted QR."""

    normalized = unit_rows(candidates)
    _, _, pivots = qr(normalized.T, mode="economic", pivoting=True)
    return normalized[pivots[:count]]


def random_orthonormal_rows(rng: np.random.Generator, count: int, width: int) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(width, count)))
    return q.T


def projection_errors(images: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    basis = orthonormal_row_basis(matrix)
    projected = (images @ basis) @ basis.T
    numerator = np.sum((images - projected) ** 2, axis=1)
    denominator = np.sum(images**2, axis=1)
    return numerator / denominator


def noisy_excess_errors(
    rng: np.random.Generator,
    images: np.ndarray,
    matrix: np.ndarray,
    noise_sigma: float,
) -> np.ndarray:
    """Error added by measurement noise beyond the noiseless row projection."""

    basis = orthonormal_row_basis(matrix)
    noiseless = (images @ basis) @ basis.T
    measurements = images @ matrix.T
    noisy = measurements + rng.normal(scale=noise_sigma, size=measurements.shape)
    recon = np.linalg.lstsq(matrix, noisy.T, rcond=1e-10)[0].T
    numerator = np.sum((recon - noiseless) ** 2, axis=1)
    denominator = np.sum(images**2, axis=1)
    return numerator / denominator


def condition_number(matrix: np.ndarray) -> float:
    singular = np.linalg.svd(matrix, compute_uv=False)
    return float(singular[0] / singular[-1])


def run(seed: int, output_dir: Path, make_plots: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    n = 32
    width = n * n

    pair_specs = [(8, 16), (12, 18), (15, 16)]
    pairs: list[PairResult] = []
    for a, b in pair_specs:
        combined = stack_lenses(n, [a, b])
        observed = numerical_rank(combined)
        predicted = pair_rank_prediction(a, b, n=n)
        pairs.append(
            PairResult(
                scales=(a, b),
                rows=int(combined.shape[0]),
                observed_rank=observed,
                predicted_rank=predicted,
                gain_over_finer_alone=observed - max(a, b) ** 2,
            )
        )

    # Complete dyadic lenses are nested.  The finer complete lens contains all
    # directions supplied by its coarser ancestors.
    dyadic_scales = [1, 2, 4, 8, 16]
    dyadic = stack_lenses(n, dyadic_scales)
    dyadic_rank = numerical_rank(dyadic)

    # A one-pixel alternating checkerboard integrates to zero in every aligned
    # dyadic cell up through scale 8 on a 32 x 32 source.
    yy, xx = np.indices((n, n))
    witness = ((xx + yy) % 2) * 2.0 - 1.0
    blind = stack_lenses(n, [1, 2, 4, 8])
    witness_pulse_max = float(np.max(np.abs(blind @ witness.reshape(-1))))
    secret_a = rng.normal(size=width)
    secret_b = secret_a + 0.75 * witness.reshape(-1)
    indistinguishable_max = float(np.max(np.abs(blind @ secret_a - blind @ secret_b)))

    # Verify that the transpose really is the adjoint gradient.
    adjoint_h = stack_lenses(n, [5, 7])
    observed = adjoint_h @ rng.normal(size=width)
    estimate = rng.normal(size=width)
    direction = rng.normal(size=width)
    direction /= np.linalg.norm(direction)
    gradient = adjoint_gradient(adjoint_h, estimate, observed)
    epsilon = 1e-6

    def loss(candidate: np.ndarray) -> float:
        residual = adjoint_h @ candidate - observed
        return 0.5 * float(residual @ residual)

    finite_difference = (loss(estimate + epsilon * direction) - loss(estimate - epsilon * direction)) / (
        2.0 * epsilon
    )
    adjoint_directional = float(gradient @ direction)
    adjoint_relative_error = float(
        abs(finite_difference - adjoint_directional)
        / max(1.0, abs(finite_difference), abs(adjoint_directional))
    )

    # The random-image null: for an isotropic image, equally ranked noiseless
    # row spaces recover the same expected fraction of energy.  The box lens is
    # given a generous geometry-only pivoted design and unit-energy rows.
    budget = 192
    trials = 256
    images = rng.normal(size=(trials, width))
    raster = np.eye(width, dtype=np.float64)[:budget]
    random_dense = random_orthonormal_rows(rng, budget, width)
    box_candidates = stack_lenses(n, [7, 11, 13, 15, 16])
    box_design = pivoted_row_design(box_candidates, budget)

    designs = {
        "pixel_raster": raster,
        "random_orthonormal": random_dense,
        "addressed_box": box_design,
    }
    random_image = {}
    noise_sigma = 0.02
    for name, matrix in designs.items():
        errors = projection_errors(images, matrix)
        excess = noisy_excess_errors(rng, images, matrix, noise_sigma)
        random_image[name] = {
            "rank": numerical_rank(matrix),
            "mean_unobserved_energy": float(np.mean(errors)),
            "sem_unobserved_energy": float(np.std(errors, ddof=1) / np.sqrt(trials)),
            "condition_number": condition_number(matrix),
            "mean_noise_excess": float(np.mean(excess)),
        }

    expected_unobserved = 1.0 - budget / width
    null_spread = float(
        max(v["mean_unobserved_energy"] for v in random_image.values())
        - min(v["mean_unobserved_energy"] for v in random_image.values())
    )

    pair_prediction_error = max(abs(p.observed_rank - p.predicted_rank) for p in pairs)
    passed = bool(
        pair_prediction_error == 0
        and dyadic_rank == 16**2
        and pairs[0].gain_over_finer_alone == 0
        and pairs[-1].gain_over_finer_alone > 0
        and witness_pulse_max < 1e-12
        and indistinguishable_max < 1e-12
        and adjoint_relative_error < 1e-7
        and all(v["rank"] == budget for v in random_image.values())
        and null_spread < 0.015
    )

    result = {
        "gate": 0,
        "name": "addressed_multiscale_observability",
        "classification": (
            "LENS_ALGEBRA_EARNED_RANDOM_IMAGE_NULL_CONFIRMED"
            if passed
            else "GATE_FAILED"
        ),
        "passed": passed,
        "seed": seed,
        "image_side": n,
        "hidden_degrees": width,
        "pairs": [asdict(pair) for pair in pairs],
        "dyadic": {
            "scales": dyadic_scales,
            "rows": int(dyadic.shape[0]),
            "rank": dyadic_rank,
            "redundant_rows": int(dyadic.shape[0] - dyadic_rank),
        },
        "nullspace_witness": {
            "scales": [1, 2, 4, 8],
            "rows": int(blind.shape[0]),
            "rank": numerical_rank(blind),
            "null_dimension": width - numerical_rank(blind),
            "witness_pulse_max_abs": witness_pulse_max,
            "two_secret_pulse_max_abs_difference": indistinguishable_max,
        },
        "software_adjoint": {
            "finite_difference_directional": float(finite_difference),
            "adjoint_directional": adjoint_directional,
            "relative_error": adjoint_relative_error,
        },
        "random_image_null": {
            "budget": budget,
            "trials": trials,
            "expected_unobserved_energy": expected_unobserved,
            "spread_between_designs": null_spread,
            "unit_energy_rows_for_fairness": True,
            "noise_sigma": noise_sigma,
            "designs": random_image,
        },
        "locked_requirements": {
            "pair_rank_prediction_error_eq_0": pair_prediction_error == 0,
            "dyadic_rank_eq_finest_rank": dyadic_rank == 16**2,
            "nested_pair_adds_no_rank": pairs[0].gain_over_finer_alone == 0,
            "incommensurate_pair_adds_rank": pairs[-1].gain_over_finer_alone > 0,
            "explicit_null_witness": witness_pulse_max < 1e-12,
            "two_distinct_secrets_same_pulses": indistinguishable_max < 1e-12,
            "adjoint_matches_finite_difference": adjoint_relative_error < 1e-7,
            "random_image_designs_rank_matched": all(
                v["rank"] == budget for v in random_image.values()
            ),
            "random_image_null_spread_lt_0_015": null_spread < 0.015,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "gate0.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    if make_plots:
        plot_results(result, secret_a, secret_b, witness.reshape(-1), blind, output_dir)
    return result


def plot_results(
    result: dict,
    secret_a: np.ndarray,
    secret_b: np.ndarray,
    witness: np.ndarray,
    blind: np.ndarray,
    output_dir: Path,
) -> None:
    n = result["image_side"]
    scales = np.arange(2, 21)
    extra = np.zeros((len(scales), len(scales)), dtype=np.float64)
    for i, a in enumerate(scales):
        for j, b in enumerate(scales):
            joint = pair_rank_prediction(int(a), int(b), n=n)
            extra[i, j] = joint - max(int(a), int(b)) ** 2

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    im = axes[0].imshow(extra, origin="lower", cmap="magma")
    axes[0].set_xticks(np.arange(0, len(scales), 3), scales[::3])
    axes[0].set_yticks(np.arange(0, len(scales), 3), scales[::3])
    axes[0].set_xlabel("second lens side")
    axes[0].set_ylabel("first lens side")
    axes[0].set_title("New rank beyond the finer lens")
    fig.colorbar(im, ax=axes[0], label="additional observable directions")

    names = list(result["random_image_null"]["designs"])
    values = [
        result["random_image_null"]["designs"][name]["mean_unobserved_energy"]
        for name in names
    ]
    sems = [
        result["random_image_null"]["designs"][name]["sem_unobserved_energy"]
        for name in names
    ]
    axes[1].bar(np.arange(len(names)), values, yerr=sems, color=["#5da9e9", "#ef8354", "#62c370"])
    axes[1].axhline(
        result["random_image_null"]["expected_unobserved_energy"],
        color="black",
        linestyle="--",
        linewidth=1,
        label="isotropic prediction",
    )
    axes[1].set_xticks(np.arange(len(names)), [name.replace("_", "\n") for name in names])
    axes[1].set_ylabel("unobserved energy fraction")
    axes[1].set_ylim(0.70, 0.82)
    axes[1].set_title("Random image: orientation cannot help")
    axes[1].legend(frameon=False)
    fig.savefig(output_dir / "gate0_rank_and_random_null.png", dpi=180)
    plt.close(fig)

    basis = orthonormal_row_basis(blind)
    observable = basis @ (basis.T @ secret_a)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), constrained_layout=True)
    panels = [secret_a, observable, witness, secret_b]
    titles = [
        "secret A",
        "what pulses identify",
        "exact invisible direction",
        "secret B = A + null",
    ]
    limits = [
        np.max(np.abs(secret_a)),
        np.max(np.abs(observable)),
        1.0,
        np.max(np.abs(secret_b)),
    ]
    for axis, panel, title, limit in zip(axes, panels, titles, limits):
        axis.imshow(panel.reshape(n, n), cmap="coolwarm", vmin=-limit, vmax=limit)
        axis.set_title(title, fontsize=10)
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle("Two different 32 x 32 worlds, exactly the same 85 scalar pulses")
    fig.savefig(output_dir / "gate0_nullspace_witness.png", dpi=180)
    plt.close(fig)


def print_receipt(result: dict) -> None:
    nested, _, coprime = result["pairs"]
    random_null = result["random_image_null"]
    print("GeometricNeuronV24 Gate 0 - addressed multiscale observability")
    print()
    print(f"hidden image degrees:                 {result['hidden_degrees']}")
    print(
        f"dyadic rows / rank:                   {result['dyadic']['rows']} / {result['dyadic']['rank']}"
    )
    print(
        f"nested lenses {tuple(nested['scales'])} rank gain:       {nested['gain_over_finer_alone']}"
    )
    print(
        f"incommensurate {tuple(coprime['scales'])} rank gain:     {coprime['gain_over_finer_alone']}"
    )
    print(
        f"explicit null dimension:              {result['nullspace_witness']['null_dimension']}"
    )
    print(
        "two-secret pulse difference:          "
        f"{result['nullspace_witness']['two_secret_pulse_max_abs_difference']:.3e}"
    )
    print(
        f"software-adjoint relative error:      {result['software_adjoint']['relative_error']:.3e}"
    )
    print(
        f"random-image expected unseen energy:  {random_null['expected_unobserved_energy']:.6f}"
    )
    for name, values in random_null["designs"].items():
        print(
            f"  {name:22s} rank {values['rank']:3d}  "
            f"unseen {values['mean_unobserved_energy']:.6f}  "
            f"cond {values['condition_number']:.3g}"
        )
    print()
    print(result["classification"])
    print(
        "The movable lens creates an addressable measurement family. Aligned dyadic "
        "scales add redundancy but no directions beyond the finest complete grid; "
        "non-nested grids add rank. A genuinely random image remains limited by rank, "
        "and distinct images in the exact nullspace are indistinguishable."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=24031976)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "gate0")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    result = run(args.seed, args.output_dir, make_plots=not args.no_plots)
    print_receipt(result)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
