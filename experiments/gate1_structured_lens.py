#!/usr/bin/env python3
"""Gate 1: does an addressed multiscale lens help on structured worlds?

Gate 0 establishes that no row-space orientation can beat another on average
for an isotropic random image at equal rank.  Gate 1 changes only the image
ensemble: smooth blobs, edges, and rectangles create repeatable covariance.

The detector still returns one scalar.  A training ensemble is used only to
choose *which known box address to read next*.  It never sees the held-out
secret.  The global PCA measurement is retained as an intentionally stronger,
non-local oracle/attacker.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnv24.lens import lens_matrix, stack_lenses  # noqa: E402


def structured_worlds(
    rng: np.random.Generator, count: int, side: int
) -> np.ndarray:
    """Generate a controlled spatial ensemble without importing a photo prior."""

    axis = (np.arange(side, dtype=np.float64) + 0.5) / side
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    images = np.zeros((count, side, side), dtype=np.float64)
    for sample in range(count):
        field = np.zeros((side, side), dtype=np.float64)
        for _ in range(int(rng.integers(1, 5))):
            cx, cy = rng.uniform(0.05, 0.95, size=2)
            sx, sy = rng.uniform(0.06, 0.28, size=2)
            amplitude = rng.normal()
            field += amplitude * np.exp(
                -0.5 * (((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2)
            )

        # A hard-edged component stops the ensemble from becoming merely a
        # low-pass Gaussian-process demonstration.
        if rng.random() < 0.8:
            y0, y1 = np.sort(rng.integers(0, side + 1, size=2))
            x0, x1 = np.sort(rng.integers(0, side + 1, size=2))
            if y1 > y0 and x1 > x0:
                field[y0:y1, x0:x1] += rng.normal(scale=0.8)

        field += rng.normal(scale=0.025, size=(side, side))
        images[sample] = field
    return images.reshape(count, side * side)


def fit_prior(train: np.ndarray, components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    centered = train - mean
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    basis = vh[:components].T
    variances = singular[:components] ** 2 / (len(train) - 1)
    return mean, basis, variances


def posterior_reconstruction(
    measurements: np.ndarray,
    matrix: np.ndarray,
    mean: np.ndarray,
    basis: np.ndarray,
    variances: np.ndarray,
    noise_sigma: float,
) -> np.ndarray:
    """Linear-Gaussian posterior mean in the learned low-rank image prior."""

    a = matrix @ basis
    covariance = (a * variances[None, :]) @ a.T
    covariance.flat[:: covariance.shape[0] + 1] += noise_sigma**2
    residual = measurements - mean @ matrix.T
    solved = np.linalg.solve(covariance, residual.T)
    latent = (variances[:, None] * a.T) @ solved
    return mean[None, :] + latent.T @ basis.T


def covariance_guided_order(
    candidates: np.ndarray,
    basis: np.ndarray,
    variances: np.ndarray,
    noise_sigma: float,
    count: int,
) -> np.ndarray:
    """Choose box addresses by greedy expected posterior variance reduction."""

    latent_rows = candidates @ basis
    covariance = np.diag(variances.copy())
    available = np.ones(len(candidates), dtype=bool)
    chosen: list[int] = []
    for _ in range(count):
        projected = latent_rows @ covariance
        predictive = np.einsum("ij,ij->i", projected, latent_rows)
        predictive[~available] = -np.inf
        # Reduction in trace(C) from a scalar Gaussian observation.
        reduction = np.einsum("ij,ij->i", projected, projected) / np.maximum(
            predictive + noise_sigma**2, 1e-15
        )
        reduction[~available] = -np.inf
        index = int(np.argmax(reduction))
        chosen.append(index)
        available[index] = False
        v = latent_rows[index]
        cv = covariance @ v
        covariance -= np.outer(cv, cv) / (noise_sigma**2 + float(v @ cv))
        covariance = 0.5 * (covariance + covariance.T)
    return np.asarray(chosen, dtype=int)


def random_address_order(
    rng: np.random.Generator, scale_offsets: list[tuple[int, int, int]], count: int
) -> np.ndarray:
    """Sample scales uniformly, then addresses uniformly, without replacement."""

    selected: list[int] = []
    selected_set: set[int] = set()
    while len(selected) < count:
        scale, start, stop = scale_offsets[int(rng.integers(len(scale_offsets)))]
        del scale
        index = int(rng.integers(start, stop))
        if index not in selected_set:
            selected.append(index)
            selected_set.add(index)
    return np.asarray(selected, dtype=int)


def relative_mse(truth: np.ndarray, estimate: np.ndarray, mean: np.ndarray) -> float:
    numerator = np.sum((truth - estimate) ** 2, axis=1)
    denominator = np.sum((truth - mean[None, :]) ** 2, axis=1)
    return float(np.mean(numerator / np.maximum(denominator, 1e-15)))


def evaluate_order(
    rng: np.random.Generator,
    matrix_order: np.ndarray,
    test: np.ndarray,
    mean: np.ndarray,
    basis: np.ndarray,
    variances: np.ndarray,
    noise_sigma: float,
    budgets: list[int],
) -> list[float]:
    errors: list[float] = []
    for budget in budgets:
        matrix = matrix_order[:budget]
        measurements = test @ matrix.T + rng.normal(
            scale=noise_sigma, size=(len(test), budget)
        )
        estimate = posterior_reconstruction(
            measurements, matrix, mean, basis, variances, noise_sigma
        )
        errors.append(relative_mse(test, estimate, mean))
    return errors


def run(seed: int, output_dir: Path, make_plots: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    side = 16
    width = side * side
    train = structured_worlds(rng, 768, side)
    test = structured_worlds(rng, 256, side)
    mean, basis, variances = fit_prior(train, components=128)
    noise_sigma = 0.03
    budgets = [8, 16, 32, 64, 96]
    max_budget = max(budgets)

    scales = list(range(1, side + 1))
    candidates = stack_lenses(side, scales)
    scale_offsets: list[tuple[int, int, int]] = []
    offset = 0
    candidate_scales = np.empty(len(candidates), dtype=int)
    for scale in scales:
        stop = offset + scale * scale
        scale_offsets.append((scale, offset, stop))
        candidate_scales[offset:stop] = scale
        offset = stop

    guided_indices = covariance_guided_order(
        candidates, basis, variances, noise_sigma, max_budget
    )
    random_indices = random_address_order(rng, scale_offsets, max_budget)

    pixel_indices = rng.permutation(width)[:max_budget]
    pixel_order = np.eye(width, dtype=np.float64)[pixel_indices]
    random_addressed_order = candidates[random_indices]
    guided_order = candidates[guided_indices]

    random_dense_order = rng.choice([-1.0, 1.0], size=(max_budget, width)) / np.sqrt(width)
    # Give the attacker independent rows with stable numerical conditioning.
    random_dense_order = np.linalg.qr(random_dense_order.T)[0].T
    pca_oracle_order = basis[:, :max_budget].T

    orders = {
        "covariance_guided_lens": guided_order,
        "random_addressed_lens": random_addressed_order,
        "fixed_fine_pixels": pixel_order,
        "random_global_masks": random_dense_order,
        "pca_global_oracle": pca_oracle_order,
    }
    errors = {
        name: evaluate_order(
            np.random.default_rng(seed + 1000 + index),
            order,
            test,
            mean,
            basis,
            variances,
            noise_sigma,
            budgets,
        )
        for index, (name, order) in enumerate(orders.items())
    }

    target_index = budgets.index(64)
    guided_64 = errors["covariance_guided_lens"][target_index]
    random_addressed_64 = errors["random_addressed_lens"][target_index]
    pixels_64 = errors["fixed_fine_pixels"][target_index]
    oracle_64 = errors["pca_global_oracle"][target_index]
    global_random_64 = errors["random_global_masks"][target_index]

    selected_scale_counts = Counter(int(candidate_scales[i]) for i in guided_indices)
    selected_scales = sorted(selected_scale_counts)
    passes = {
        "guided_beats_random_addressed_by_10pct": guided_64 <= 0.90 * random_addressed_64,
        "guided_beats_fixed_pixels_by_10pct": guided_64 <= 0.90 * pixels_64,
        "global_oracle_remains_upper_bound": oracle_64 <= guided_64,
        "more_than_one_lens_scale_selected": len(selected_scales) > 1,
    }
    passed = all(passes.values())
    result = {
        "gate": 1,
        "name": "structured_world_address_policy",
        "classification": (
            "ADDRESS_POLICY_HELPS_STRUCTURED_WORLDS_BUT_GLOBAL_ORACLE_WINS"
            if passed
            else "STRUCTURED_LENS_GATE_FAILED"
        ),
        "passed": passed,
        "seed": seed,
        "image_side": side,
        "train_worlds": len(train),
        "test_worlds": len(test),
        "prior_components": basis.shape[1],
        "noise_sigma": noise_sigma,
        "budgets": budgets,
        "relative_mse": errors,
        "budget_64": {
            "guided": guided_64,
            "random_addressed": random_addressed_64,
            "fixed_fine_pixels": pixels_64,
            "random_global_masks": global_random_64,
            "pca_global_oracle": oracle_64,
            "guided_gain_vs_random_addressed": 1.0 - guided_64 / random_addressed_64,
            "guided_gain_vs_fixed_pixels": 1.0 - guided_64 / pixels_64,
        },
        "guided_first_96_scale_counts": {
            str(key): selected_scale_counts[key] for key in selected_scales
        },
        "locked_requirements": passes,
        "scope": (
            "The selector knows the training ensemble covariance and the lens family, "
            "but never sees a held-out secret before selecting its addresses."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "gate1.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if make_plots:
        plot_results(result, test, mean, orders, basis, variances, output_dir, seed)
    return result


def plot_results(
    result: dict,
    test: np.ndarray,
    mean: np.ndarray,
    orders: dict[str, np.ndarray],
    basis: np.ndarray,
    variances: np.ndarray,
    output_dir: Path,
    seed: int,
) -> None:
    colors = {
        "covariance_guided_lens": "#52b788",
        "random_addressed_lens": "#adb5bd",
        "fixed_fine_pixels": "#4ea8de",
        "random_global_masks": "#f4a261",
        "pca_global_oracle": "#9b5de5",
    }
    fig, axis = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    for name, values in result["relative_mse"].items():
        axis.plot(
            result["budgets"],
            values,
            marker="o",
            linewidth=2,
            color=colors[name],
            label=name.replace("_", " "),
        )
    axis.set_xlabel("scalar pulses")
    axis.set_ylabel("held-out relative MSE")
    axis.set_title("A movable local lens helps only after the world has structure")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize=8)
    fig.savefig(output_dir / "gate1_policy_curves.png", dpi=180)
    plt.close(fig)

    sample = test[3:4]
    budget = 64
    names = [
        "covariance_guided_lens",
        "random_addressed_lens",
        "fixed_fine_pixels",
        "random_global_masks",
        "pca_global_oracle",
    ]
    reconstructions = []
    for index, name in enumerate(names):
        matrix = orders[name][:budget]
        local_rng = np.random.default_rng(seed + 9000 + index)
        measurements = sample @ matrix.T + local_rng.normal(
            scale=result["noise_sigma"], size=(1, budget)
        )
        reconstructions.append(
            posterior_reconstruction(
                measurements,
                matrix,
                mean,
                basis,
                variances,
                result["noise_sigma"],
            )[0]
        )

    side = result["image_side"]
    panels = [sample[0], *reconstructions]
    titles = ["secret", "guided lens", "random lens", "pixels", "random global", "PCA oracle"]
    limit = max(float(np.max(np.abs(panel))) for panel in panels)
    fig, axes = plt.subplots(1, len(panels), figsize=(13.5, 2.7), constrained_layout=True)
    for axis, panel, title in zip(axes, panels, titles):
        axis.imshow(panel.reshape(side, side), cmap="coolwarm", vmin=-limit, vmax=limit)
        axis.set_title(title, fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle("One held-out structured world after 64 scalar measurements")
    fig.savefig(output_dir / "gate1_reconstructions.png", dpi=180)
    plt.close(fig)


def print_receipt(result: dict) -> None:
    values = result["budget_64"]
    print("GeometricNeuronV24 Gate 1 - structured-world address policy")
    print()
    print(f"train / held-out worlds:              {result['train_worlds']} / {result['test_worlds']}")
    print(f"scalar pulse budget:                  64")
    print(f"covariance-guided local lens MSE:     {values['guided']:.6f}")
    print(f"random addressed local lens MSE:     {values['random_addressed']:.6f}")
    print(f"fixed fine-pixel MSE:                 {values['fixed_fine_pixels']:.6f}")
    print(f"random global-mask MSE:               {values['random_global_masks']:.6f}")
    print(f"PCA global oracle MSE:                {values['pca_global_oracle']:.6f}")
    print(
        f"guided gain vs random address:        {100 * values['guided_gain_vs_random_addressed']:.2f}%"
    )
    print(
        f"guided gain vs fixed pixels:          {100 * values['guided_gain_vs_fixed_pixels']:.2f}%"
    )
    print(f"guided scales used:                   {result['guided_first_96_scale_counts']}")
    print()
    print(result["classification"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=24031976)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "gate1")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    result = run(args.seed, args.output_dir, make_plots=not args.no_plots)
    print_receipt(result)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
