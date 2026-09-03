#!/usr/bin/env python3
"""Post-result audit for Gate 4.

The original Gate-4 success condition included >=0.90 exact identification of
which of 12 local 10%-leak changes occurred under 1 microvolt RMS soma noise.
The first run did not meet it.  This audit does not lower that threshold.

Instead it asks two narrower post-result questions:
1. Is the algebraic observability gain from addressed stimulation stable?
2. At what soma-noise scale does exact hidden-region identity become reliable?

The sensitivity matrix is built once on the same pinned cell-1125 morphology.
Only observation-noise seeds and random probe subsets are varied.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gate4_cell1125_observability import (
    CLUSTER_RADII_UM,
    HIDDEN_PARAMS,
    NOISE_SIGMA_MV,
    PROBE_BUDGET,
    ROOT,
    build_probes,
    cable_graph,
    choose_hidden_sections,
    choose_probe_centers,
    download_source,
    fingerprint_accuracy,
    greedy_logdet,
    load_morphio_tree,
    random_audit,
    section_metadata,
    sensitivity_matrix,
    singular_metrics,
    SOURCE_NAME,
)


NOISE_SWEEP_MV = (0.0001, 0.00025, 0.0005, 0.001, 0.002, 0.005)


def summarize(values: list[float]) -> dict:
    x = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "std": float(np.std(x)),
        "per_seed": x.tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "gate4_robustness",
    )
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source = download_source(args.output_dir / "_source" / SOURCE_NAME)
    tree = load_morphio_tree(source)
    graph = cable_graph(tree)
    sections = section_metadata(tree, graph)
    hidden = choose_hidden_sections(sections)
    centers = choose_probe_centers(sections)
    probes, meta = build_probes(tree, graph, centers)
    J, _ = sensitivity_matrix(tree, graph, hidden, probes)

    all_indices = np.arange(len(meta), dtype=int)
    point_indices = np.asarray(
        [i for i, row in enumerate(meta) if row["radius_um"] == 0.0],
        dtype=int,
    )
    active = greedy_logdet(J, all_indices, PROBE_BUDGET)
    active_point = greedy_logdet(J, point_indices, PROBE_BUDGET)
    active_metrics = singular_metrics(J[active])
    point_metrics = singular_metrics(J[active_point])

    random = random_audit(J, all_indices, PROBE_BUDGET, seed=40404)

    seeds = list(range(1, 11))
    acc_1uv = [
        fingerprint_accuracy(
            J,
            active,
            seed=9000 + seed,
            trials=8192,
            noise_sigma_mV=NOISE_SIGMA_MV,
        )
        for seed in seeds
    ]
    point_acc_1uv = [
        fingerprint_accuracy(
            J,
            active_point,
            seed=10000 + seed,
            trials=8192,
            noise_sigma_mV=NOISE_SIGMA_MV,
        )
        for seed in seeds
    ]

    noise_sweep = {}
    for sigma in NOISE_SWEEP_MV:
        vals = [
            fingerprint_accuracy(
                J,
                active,
                seed=11000 + 100 * si + k,
                trials=4096,
                noise_sigma_mV=sigma,
            )
            for si, k in enumerate(seeds)
        ]
        noise_sweep[str(sigma)] = summarize(vals)

    rng = np.random.default_rng(50505)
    random_fingerprint = []
    for draw in range(32):
        selected = rng.choice(
            all_indices,
            size=PROBE_BUDGET,
            replace=False,
        ).tolist()
        random_fingerprint.append(
            fingerprint_accuracy(
                J,
                selected,
                seed=12000 + draw,
                trials=2048,
                noise_sigma_mV=NOISE_SIGMA_MV,
            )
        )

    strict_pass_count = int(np.sum(np.asarray(acc_1uv) >= 0.90))
    algebraic_mechanism = bool(
        active_metrics["numerical_rank"] == HIDDEN_PARAMS
        and active_metrics["numerical_rank"] > random["numerical_rank"]["median"]
        and active_metrics["smallest_singular_mV"]
        > random["smallest_singular_mV"]["p90"]
        and point_metrics["numerical_rank"] == HIDDEN_PARAMS
    )
    identity_gate_replicated = strict_pass_count == len(seeds)

    if algebraic_mechanism and not identity_gate_replicated:
        classification = "ADDRESS_OPENS_FULL_RANK_BUT_SOMA_NOISE_LIMITS_IDENTITY"
    elif algebraic_mechanism and identity_gate_replicated:
        classification = "ADDRESS_OPENS_FULL_RANK_AND_IDENTITY_REPLICATES"
    else:
        classification = "ADDRESS_OBSERVABILITY_MECHANISM_NOT_ROBUST"

    result = {
        "audit": "gate4_robustness_and_noise_boundary",
        "classification": classification,
        "algebraic_mechanism_replicated": algebraic_mechanism,
        "original_1uv_identity_gate_replicated": identity_gate_replicated,
        "original_1uv_pass_count": strict_pass_count,
        "original_1uv_total": len(seeds),
        "active_multiscale": {
            "metrics": active_metrics,
            "accuracy_at_1uv": summarize(acc_1uv),
            "selected_scale_counts": {
                str(radius): int(
                    sum(abs(meta[i]["radius_um"] - radius) < 1e-12 for i in active)
                )
                for radius in CLUSTER_RADII_UM
            },
        },
        "active_point_only": {
            "metrics": point_metrics,
            "accuracy_at_1uv": summarize(point_acc_1uv),
        },
        "random_multiscale": {
            "singular_audit": random,
            "fingerprint_accuracy_at_1uv_32_subsets": summarize(random_fingerprint),
        },
        "noise_sweep_active_multiscale": noise_sweep,
        "note": (
            "The >=0.90 requirement at 0.001 mV RMS noise remains the original "
            "Gate-4 criterion. The sweep was added only after that criterion failed."
        ),
    }

    (args.output_dir / "robustness.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )

    print("GeometricNeuronV24 Gate 4 post-result audit")
    print()
    print(
        "active rank / random median rank:     "
        f"{active_metrics['numerical_rank']} / {random['numerical_rank']['median']:.1f}"
    )
    print(
        "active s_min / random p90 s_min:      "
        f"{active_metrics['smallest_singular_mV']:.4e} / "
        f"{random['smallest_singular_mV']['p90']:.4e} mV"
    )
    print(
        "1 uV active accuracy mean/min/max:    "
        f"{result['active_multiscale']['accuracy_at_1uv']['mean']:.3f} / "
        f"{result['active_multiscale']['accuracy_at_1uv']['min']:.3f} / "
        f"{result['active_multiscale']['accuracy_at_1uv']['max']:.3f}"
    )
    print(
        "1 uV random-subset accuracy mean:     "
        f"{result['random_multiscale']['fingerprint_accuracy_at_1uv_32_subsets']['mean']:.3f}"
    )
    print(
        "original >=0.90 gate passes:          "
        f"{strict_pass_count} / {len(seeds)}"
    )
    print("noise sweep:")
    for sigma in NOISE_SWEEP_MV:
        row = noise_sweep[str(sigma)]
        print(
            f"  {1000*sigma:5.2f} uV RMS -> "
            f"{row['mean']:.3f} mean accuracy "
            f"(min {row['min']:.3f})"
        )
    print()
    print(classification)
    raise SystemExit(0 if algebraic_mechanism else 1)


if __name__ == "__main__":
    main()
