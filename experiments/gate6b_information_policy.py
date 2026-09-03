#!/usr/bin/env python3
"""Gate 6B post-result audit: why multiscale failed, and what policy uses it.

Gate 6's persistent-write mechanism passed strongly, but its locked multiscale
criterion failed: the raw predictive-variance policy used 4.38 probes per
appearance, exactly the same as the fine-only policy.

That failure is preserved.

This post-result audit changes only the *probe-selection score*.  Instead of
maximizing raw variance of predicted scalar amplitudes, it maximizes expected
Bayesian information gain (mutual information between the hidden prototype
identity and one noisy scalar measurement).

Why this matters here:
- a fine 4x4 lens has a huge signal but asks an unbalanced one-vs-seven question;
- a 16x16 lens has a diluted signal but asks a balanced four-vs-four question;
- both are far above detector noise.

Raw variance over-rewards amplitude. Information gain values the partition.

This is explicitly a post-result policy audit, not a rewrite of the original
Gate-6 preregistration.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

import gate6_perceptionlab_memory as g6


GRID_POINTS = 401
GRID_SIGMAS = 7.0
MIN_COST_RATIO_TARGET = 0.80
MIN_LOCALIZATION_ACCURACY = 0.99


def entropy_bits(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def expected_information_bits(
    posterior: np.ndarray,
    means: np.ndarray,
    sigma: float,
) -> float:
    """Numerically integrate I(H;Y) for a finite Gaussian-mixture channel."""
    p = np.asarray(posterior, dtype=float)
    mu = np.asarray(means, dtype=float)
    sigma = float(sigma)

    if len(mu) == 0 or sigma <= 0:
        return 0.0

    lo = float(np.min(mu) - GRID_SIGMAS * sigma)
    hi = float(np.max(mu) + GRID_SIGMAS * sigma)
    if hi <= lo:
        return 0.0

    y = np.linspace(lo, hi, GRID_POINTS)
    z = (y[None, :] - mu[:, None]) / sigma
    phi = np.exp(-0.5 * z * z) / (np.sqrt(2.0 * np.pi) * sigma)
    joint = p[:, None] * phi
    mixture = np.sum(joint, axis=0)
    safe = np.maximum(mixture, 1e-300)
    post_y = joint / safe[None, :]

    h_y = -np.sum(
        np.where(post_y > 0, post_y * np.log2(np.maximum(post_y, 1e-300)), 0.0),
        axis=0,
    )
    expected_h = float(np.trapezoid(mixture * h_y, y))
    return max(0.0, entropy_bits(p) - expected_h)


def search_onset_information(
    world: np.ndarray,
    memory: np.ndarray,
    t: int,
    templates: list[np.ndarray],
    lenses: list[g6.Lens],
    policy: str,
    rng: np.random.Generator,
) -> dict:
    posterior = np.full(len(templates), 1.0 / len(templates), dtype=float)
    unused = list(range(len(lenses)))
    chosen: list[dict] = []

    for _ in range(g6.MAX_SEARCH_PROBES):
        candidates = list(unused)
        if not candidates:
            break

        pred_by_lens = []
        scores = []
        for li in candidates:
            lens = lenses[li]
            mu = np.asarray(
                [
                    g6.lens_read(g6.shift_world(template, t), lens, t)
                    for template in templates
                ],
                dtype=float,
            )
            pred_by_lens.append(mu)
            scores.append(
                expected_information_bits(
                    posterior,
                    mu,
                    g6.PAID_NOISE_SIGMA,
                )
            )

        local_index = int(np.argmax(scores))
        lens_index = int(candidates[local_index])
        lens = lenses[lens_index]
        mu = pred_by_lens[local_index]

        observed_residual = (
            g6.lens_read(world, lens, t)
            - g6.lens_read(memory, lens, t)
            + float(rng.normal(scale=g6.PAID_NOISE_SIGMA))
        )

        logp = (
            np.log(np.maximum(posterior, 1e-300))
            - 0.5
            * ((observed_residual - mu) / g6.PAID_NOISE_SIGMA) ** 2
        )
        logp -= float(np.max(logp))
        posterior = np.exp(logp)
        posterior /= float(np.sum(posterior))

        chosen.append(
            {
                "lens": lens.name,
                "scale": int(lens.scale),
                "observed_residual": float(observed_residual),
                "posterior_max": float(np.max(posterior)),
                "expected_information_bits": float(scores[local_index]),
            }
        )
        unused.remove(lens_index)

        if float(np.max(posterior)) >= g6.POSTERIOR_STOP:
            break

    return {
        "identity": int(np.argmax(posterior)),
        "confidence": float(np.max(posterior)),
        "probes": int(len(chosen)),
        "trace": chosen,
    }


def aggregate(rows: list[dict]) -> dict:
    def mean(key: str) -> float:
        return float(np.mean([float(r[key]) for r in rows]))

    return {
        "paid_probes_mean": mean("paid_probes"),
        "pre_mse_mean": mean("mean_pre_prediction_mse"),
        "onset_search_probes_mean": mean("mean_onset_search_probes"),
        "localization_accuracy_mean": mean("onset_localization_accuracy"),
        "repeated_on_paid_steps_mean": mean("repeated_on_paid_steps"),
    }


def scale_sequence_counts(rows: list[dict]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        for seq in row["onset_search_scales"]:
            counter["-".join(str(int(x)) for x in seq)] += 1
    return dict(counter.most_common())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=g6.ROOT / "results" / "gate6b",
    )
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Run the original policies before monkey-patching the selector.
    original_variance = [
        g6.simulate(seed, "ACTIVE_WRITE_MULTISCALE")
        for seed in g6.AUDIT_SEEDS
    ]
    fine_only = [
        g6.simulate(seed, "ACTIVE_WRITE_FINE")
        for seed in g6.AUDIT_SEEDS
    ]

    original_search = g6.search_onset
    try:
        g6.search_onset = search_onset_information
        information = [
            g6.simulate(seed, "ACTIVE_WRITE_MULTISCALE")
            for seed in g6.AUDIT_SEEDS
        ]
    finally:
        g6.search_onset = original_search

    a_var = aggregate(original_variance)
    a_fine = aggregate(fine_only)
    a_info = aggregate(information)

    info_vs_fine = (
        a_info["onset_search_probes_mean"]
        / max(a_fine["onset_search_probes_mean"], 1e-30)
    )
    info_vs_variance = (
        a_info["onset_search_probes_mean"]
        / max(a_var["onset_search_probes_mean"], 1e-30)
    )

    requirements = {
        "information_policy_localization_ge_0p99": (
            a_info["localization_accuracy_mean"] >= MIN_LOCALIZATION_ACCURACY
        ),
        "information_policy_onset_cost_le_0p80_fine": (
            info_vs_fine <= MIN_COST_RATIO_TARGET
        ),
        "information_policy_beats_raw_variance": (
            info_vs_variance <= MIN_COST_RATIO_TARGET
        ),
        "persistent_memory_quality_preserved": (
            a_info["pre_mse_mean"] <= 1.05 * a_var["pre_mse_mean"]
        ),
    }
    passed = all(requirements.values())

    classification = (
        "INFORMATION_GAIN_UNLOCKS_COARSE_TO_FINE_SEARCH_POSTHOC"
        if passed
        else "INFORMATION_GAIN_DOES_NOT_RESCUE_MULTISCALE"
    )

    result = {
        "audit": "gate6b_post_result_information_policy",
        "classification": classification,
        "passed": passed,
        "original_gate6_multiscale_requirement_remains_failed": True,
        "policies": {
            "raw_predictive_variance_multiscale": a_var,
            "fine_only": a_fine,
            "expected_information_multiscale": a_info,
        },
        "comparisons": {
            "information_vs_fine_onset_cost_ratio": float(info_vs_fine),
            "information_vs_raw_variance_onset_cost_ratio": float(info_vs_variance),
        },
        "information_policy_scale_sequences": scale_sequence_counts(information),
        "raw_variance_scale_sequences": scale_sequence_counts(original_variance),
        "requirements": requirements,
        "note": (
            "This selector was introduced only after Gate 6 showed that raw "
            "predictive variance did not exploit the available lens scales."
        ),
    }

    (args.output_dir / "gate6b.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )

    print("GeometricNeuronV24 Gate 6B - post-result information policy")
    print()
    print(
        "raw variance onset probes:       "
        f"{a_var['onset_search_probes_mean']:.3f}"
    )
    print(
        "fine-only onset probes:          "
        f"{a_fine['onset_search_probes_mean']:.3f}"
    )
    print(
        "information-gain onset probes:   "
        f"{a_info['onset_search_probes_mean']:.3f}"
    )
    print(
        "info / fine ratio:               "
        f"{info_vs_fine:.3f}"
    )
    print(
        "info / raw-variance ratio:       "
        f"{info_vs_variance:.3f}"
    )
    print(
        "information localization:        "
        f"{a_info['localization_accuracy_mean']:.3f}"
    )
    print(
        "top information scale sequence: "
        f"{next(iter(result['information_policy_scale_sequences'].items()), ('none', 0))}"
    )
    print()
    print(classification)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
