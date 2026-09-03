#!/usr/bin/env python3
"""Gate 5: does HUMAN NMDA rescue weak soma-observability directions?

Gate 4 found a clean but incomplete bridge to a real released morphology:
addressed dendritic current probes opened a 12-parameter passive soma
observability Jacobian to full numerical rank, but only 7/12 singular
directions cleared a locked 1-uV RMS soma-noise ruler.

Gate 5 keeps the Gate-4 morphology, hidden leak perturbations, probe family,
probe budget, soma readout, and *the exact 12 probes selected by the passive
Gate-4 policy*.  Only the local input law changes.

The four conditions are:

    PASSIVE_CURRENT
        fixed 0.10 nA current, exactly Gate 4.

    AMPA_ONLY
        voltage-dependent driving force, rest-matched to 0.10 nA.

    FROZEN_NMDA
        HUMAN AMPA/NMDA conductance ratio, but magnesium block frozen at
        Vrest.  It is rest-matched and therefore algebraically identical to
        AMPA_ONLY in this static assay.  This is a deliberate control.

    HUMAN_NMDA
        same raw AMPA/NMDA ratio and same current at Vrest, but with the
        released HUMAN Jahr-Stevens magnesium block gamma=0.078 /mV.

For conductance conditions, depolarization x relative to Vrest=-70 mV solves

    A x = I_syn(x)

and hidden parameter p_j is the same 10% local leak-density increase used in
Gate 4.  Implicit differentiation gives

    (A - dI/dx) dx/dp_j = -D_j x.

The primary question is deliberately narrow:

    On the *same passive-selected addresses*, does HUMAN magnesium feedback
    selectively amplify the weak Gate-4 hidden-parameter directions enough to
    improve soma observability beyond a rest-matched frozen-block control?

This is still a reduced morphology-graph model, not the released FCI NEURON
cell and not evidence that a biological neuron performs tomography on itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from gate4_cell1125_observability import (
    HIDDEN_PARAMS,
    NOISE_SIGMA_MV,
    PROBE_BUDGET,
    PROBE_CURRENT_NA,
    ROOT,
    SOURCE_NAME,
    build_probes,
    cable_graph,
    choose_hidden_sections,
    choose_probe_centers,
    download_source,
    fingerprint_accuracy,
    greedy_logdet,
    hidden_derivative_matrix,
    load_morphio_tree,
    section_metadata,
    sensitivity_matrix,
    singular_metrics,
)


V_REST_MV = -70.0
E_SYN_MV = 0.0
HUMAN_GAMMA_PER_MV = 0.078
G_AMPA_SOURCE_US = 0.00088
G_NMDA_SOURCE_US = 0.00131
RAW_NMDA_TO_AMPA = G_NMDA_SOURCE_US / G_AMPA_SOURCE_US

N_ACCURACY_SEEDS = 10
ACCURACY_TRIALS = 8192

# These are locked before the first Gate-5 run.
RESCUE_MIN_VISIBLE_RANK_GAIN = 1
RESCUE_MIN_ACCURACY_GAIN = 0.05
RESCUE_MIN_WEAK_GAIN = 1.20
RESCUE_MIN_WEAK_SELECTIVITY = 1.10
FD_MAX_RELATIVE_ERROR = 0.08
FROZEN_AMPA_EQUIVALENCE_ATOL_MV = 1e-12


def nmda_block(v_abs_mV: np.ndarray) -> np.ndarray:
    v = np.asarray(v_abs_mV, dtype=float)
    return 1.0 / (
        1.0 + np.exp(-HUMAN_GAMMA_PER_MV * v) * (1.0 / 3.57)
    )


def nmda_block_derivative(v_abs_mV: np.ndarray) -> np.ndarray:
    b = nmda_block(v_abs_mV)
    return HUMAN_GAMMA_PER_MV * b * (1.0 - b)


def rest_matched_conductances(
    weights: np.ndarray,
    condition: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-node raw AMPA and NMDA conductances in uS.

    Every conductance condition produces exactly PROBE_CURRENT_NA at Vrest
    before cable voltage changes.  weights sum to one.
    """
    w = np.asarray(weights, dtype=float)
    if abs(float(np.sum(w)) - 1.0) > 1e-9:
        raise ValueError("probe weights must sum to one")

    total_effective_uS = PROBE_CURRENT_NA / (E_SYN_MV - V_REST_MV)

    if condition == "AMPA_ONLY":
        return total_effective_uS * w, np.zeros_like(w)

    if condition in ("FROZEN_NMDA", "HUMAN_NMDA"):
        b0 = float(nmda_block(np.asarray([V_REST_MV]))[0])
        g_ampa_total = total_effective_uS / (
            1.0 + RAW_NMDA_TO_AMPA * b0
        )
        g_nmda_total = RAW_NMDA_TO_AMPA * g_ampa_total
        return g_ampa_total * w, g_nmda_total * w

    raise ValueError(condition)


def synaptic_current_and_derivative(
    x_mV: np.ndarray,
    weights: np.ndarray,
    condition: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return inward current I(x) [nA] and dI/dx [uS]."""
    x = np.asarray(x_mV, dtype=float)

    if condition == "PASSIVE_CURRENT":
        return PROBE_CURRENT_NA * np.asarray(weights, dtype=float), np.zeros_like(x)

    g_a, g_n = rest_matched_conductances(weights, condition)
    driving_mV = E_SYN_MV - (V_REST_MV + x)

    if condition == "AMPA_ONLY":
        current = g_a * driving_mV
        deriv = -g_a
        return current, deriv

    if condition == "FROZEN_NMDA":
        b0 = float(nmda_block(np.asarray([V_REST_MV]))[0])
        g_eff = g_a + g_n * b0
        current = g_eff * driving_mV
        deriv = -g_eff
        return current, deriv

    if condition == "HUMAN_NMDA":
        v_abs = V_REST_MV + x
        b = nmda_block(v_abs)
        db = nmda_block_derivative(v_abs)
        g_eff = g_a + g_n * b
        current = g_eff * driving_mV
        deriv = g_n * db * driving_mV - g_eff
        return current, deriv

    raise ValueError(condition)


def solve_equilibrium(
    A: sparse.csc_matrix,
    weights: np.ndarray,
    condition: str,
    *,
    base_lu=None,
    max_iter: int = 400,
    tol_mV: float = 1e-11,
) -> dict:
    """Solve A x = I_syn(x) with a cheap fixed-point/Newton fallback."""
    if base_lu is None:
        base_lu = splu(A)

    w = np.asarray(weights, dtype=float)

    if condition == "PASSIVE_CURRENT":
        current = PROBE_CURRENT_NA * w
        x = base_lu.solve(current)
        return {
            "x_mV": x,
            "current_nA": current,
            "dI_dx_uS": np.zeros_like(x),
            "iterations": 1,
            "residual_mV_uS": 0.0,
            "converged": True,
            "solver": "linear",
        }

    # Rest-matched current is a stable initial guess.
    x = base_lu.solve(PROBE_CURRENT_NA * w)
    converged = False
    residual = np.inf
    solver = "fixed_point"

    for it in range(max_iter):
        current, deriv = synaptic_current_and_derivative(x, w, condition)
        target = base_lu.solve(current)
        trial = 0.55 * x + 0.45 * target
        residual = float(np.max(np.abs(trial - x)))
        x = trial
        if residual < tol_mV:
            converged = True
            break

    if not converged:
        solver = "newton"
        for nit in range(40):
            current, deriv = synaptic_current_and_derivative(x, w, condition)
            F = A @ x - current
            residual = float(np.max(np.abs(F)))
            if residual < 1e-10:
                converged = True
                break
            J = (A - sparse.diags(deriv, format="csc")).tocsc()
            step = splu(J).solve(-F)
            scale = 1.0
            base_norm = float(np.max(np.abs(F)))
            accepted = False
            for _ in range(14):
                trial = x + scale * step
                tcurrent, _ = synaptic_current_and_derivative(
                    trial, w, condition
                )
                tres = float(np.max(np.abs(A @ trial - tcurrent)))
                if tres < base_norm:
                    x = trial
                    accepted = True
                    break
                scale *= 0.5
            if not accepted:
                break
        it = max_iter + nit + 1

    current, deriv = synaptic_current_and_derivative(x, w, condition)
    final_residual = float(np.max(np.abs(A @ x - current)))
    converged = bool(converged or final_residual < 1e-9)

    return {
        "x_mV": x,
        "current_nA": current,
        "dI_dx_uS": deriv,
        "iterations": int(it + 1),
        "residual_mV_uS": final_residual,
        "converged": converged,
        "solver": solver,
    }


def nonlinear_sensitivity_row(
    graph: dict,
    hidden_D: np.ndarray,
    weights: np.ndarray,
    condition: str,
    *,
    base_lu,
) -> tuple[np.ndarray, dict]:
    A = graph["A_uS"]
    solved = solve_equilibrium(
        A,
        weights,
        condition,
        base_lu=base_lu,
    )
    if not solved["converged"]:
        raise RuntimeError(f"{condition} equilibrium did not converge")

    x = solved["x_mV"]
    deriv = solved["dI_dx_uS"]
    jac = (A - sparse.diags(deriv, format="csc")).tocsc()
    jac_lu = splu(jac)

    # p_j=1 is the same 10% section leak change as Gate 4.
    rhs = -(hidden_D * x[:, None])
    dxdp = jac_lu.solve(rhs)
    row = np.asarray(dxdp[0, :], dtype=float)

    meta = {
        "soma_depolarization_mV": float(x[0]),
        "max_local_depolarization_mV": float(np.max(x)),
        "min_local_depolarization_mV": float(np.min(x)),
        "total_synaptic_current_nA": float(np.sum(solved["current_nA"])),
        "max_dI_dx_uS": float(np.max(deriv)),
        "min_dI_dx_uS": float(np.min(deriv)),
        "solver": solved["solver"],
        "iterations": solved["iterations"],
        "residual": solved["residual_mV_uS"],
    }
    return row, meta


def matrix_for_selected(
    graph: dict,
    hidden_D: np.ndarray,
    normalized_weights: np.ndarray,
    selected: list[int],
    condition: str,
) -> tuple[np.ndarray, list[dict]]:
    base_lu = splu(graph["A_uS"])
    rows = []
    meta = []
    for probe_index in selected:
        row, info = nonlinear_sensitivity_row(
            graph,
            hidden_D,
            normalized_weights[:, probe_index],
            condition,
            base_lu=base_lu,
        )
        rows.append(row)
        meta.append({"probe_index": int(probe_index), **info})
    return np.stack(rows, axis=0), meta


def paired_accuracy_summary(
    J: np.ndarray,
    selected_local_rows: list[int],
    *,
    seed_base: int,
) -> dict:
    values = [
        fingerprint_accuracy(
            J,
            selected_local_rows,
            seed=seed_base + k,
            trials=ACCURACY_TRIALS,
            noise_sigma_mV=NOISE_SIGMA_MV,
        )
        for k in range(N_ACCURACY_SEEDS)
    ]
    x = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "std": float(np.std(x)),
        "per_seed": x.tolist(),
    }


def directional_norms(J: np.ndarray, directions: np.ndarray) -> np.ndarray:
    # directions: hidden_parameter x k
    return np.linalg.norm(
        np.asarray(J, dtype=float) @ np.asarray(directions, dtype=float),
        axis=0,
    )


def finite_difference_check(
    graph: dict,
    hidden_D: np.ndarray,
    weights: np.ndarray,
    condition: str,
    analytic_row: np.ndarray,
    parameter_index: int,
) -> dict:
    A = graph["A_uS"]
    base = solve_equilibrium(
        A,
        weights,
        condition,
        base_lu=splu(A),
    )
    if not base["converged"]:
        raise RuntimeError("baseline finite-difference solve failed")

    delta_diag = hidden_D[:, parameter_index]
    A_pert = (A + sparse.diags(delta_diag, format="csc")).tocsc()
    pert = solve_equilibrium(
        A_pert,
        weights,
        condition,
        base_lu=splu(A_pert),
    )
    if not pert["converged"]:
        raise RuntimeError("perturbed finite-difference solve failed")

    exact = float(pert["x_mV"][0] - base["x_mV"][0])
    linear = float(analytic_row[parameter_index])
    rel = abs(exact - linear) / (abs(exact) + 1e-30)
    return {
        "condition": condition,
        "parameter_index": int(parameter_index),
        "exact_delta_mV": exact,
        "analytic_delta_mV": linear,
        "relative_error": float(rel),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "gate5",
    )
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source = download_source(args.output_dir / "_source" / SOURCE_NAME)
    tree = load_morphio_tree(source)
    graph = cable_graph(tree)
    sections = section_metadata(tree, graph)
    hidden = choose_hidden_sections(sections)
    centers = choose_probe_centers(sections)
    probes, probe_meta = build_probes(tree, graph, centers)

    # Recreate Gate 4 exactly and freeze its chosen addresses before applying
    # any nonlinear input law.
    J_passive_all, _ = sensitivity_matrix(
        tree,
        graph,
        hidden,
        probes,
    )
    all_indices = np.arange(len(probe_meta), dtype=int)
    selected = greedy_logdet(
        J_passive_all,
        all_indices,
        PROBE_BUDGET,
    )
    J_passive = J_passive_all[selected]

    normalized_weights = probes / PROBE_CURRENT_NA
    hidden_D = hidden_derivative_matrix(tree, graph, hidden)

    matrices = {"PASSIVE_CURRENT": J_passive}
    run_meta = {"PASSIVE_CURRENT": []}
    for condition in ("AMPA_ONLY", "FROZEN_NMDA", "HUMAN_NMDA"):
        J, meta = matrix_for_selected(
            graph,
            hidden_D,
            normalized_weights,
            selected,
            condition,
        )
        matrices[condition] = J
        run_meta[condition] = meta

    # Sanity: PASSIVE_CURRENT through the nonlinear helper must reproduce Gate 4.
    J_passive_helper, passive_helper_meta = matrix_for_selected(
        graph,
        hidden_D,
        normalized_weights,
        selected,
        "PASSIVE_CURRENT",
    )
    passive_reproduction_error = float(
        np.max(np.abs(J_passive_helper - J_passive))
    )

    # Under rest matching, AMPA_ONLY and FROZEN_NMDA are intentionally the
    # exact same static conductance law. Any discrepancy is a bug.
    frozen_ampa_max_abs = float(
        np.max(np.abs(matrices["AMPA_ONLY"] - matrices["FROZEN_NMDA"]))
    )

    metrics = {
        condition: singular_metrics(J)
        for condition, J in matrices.items()
    }

    local_rows = list(range(PROBE_BUDGET))
    accuracy = {
        condition: paired_accuracy_summary(
            J,
            local_rows,
            seed_base=51000,
        )
        for condition, J in matrices.items()
    }

    # Gate-4 weak directions are defined from the passive SVD, not from the
    # HUMAN result. This prevents outcome-dependent subspace selection.
    _, s_passive, vh_passive = np.linalg.svd(
        J_passive,
        full_matrices=False,
    )
    passive_visible = metrics["PASSIVE_CURRENT"]["noise_visible_rank"]
    weak_count = max(1, HIDDEN_PARAMS - passive_visible)
    weak_dirs = vh_passive.T[:, -weak_count:]
    strong_dirs = vh_passive.T[:, : HIDDEN_PARAMS - weak_count]

    directional = {}
    for condition, J in matrices.items():
        directional[condition] = {
            "weak_norms_mV": directional_norms(J, weak_dirs).tolist(),
            "strong_norms_mV": directional_norms(J, strong_dirs).tolist(),
        }

    frozen_weak = np.asarray(
        directional["FROZEN_NMDA"]["weak_norms_mV"],
        dtype=float,
    )
    human_weak = np.asarray(
        directional["HUMAN_NMDA"]["weak_norms_mV"],
        dtype=float,
    )
    frozen_strong = np.asarray(
        directional["FROZEN_NMDA"]["strong_norms_mV"],
        dtype=float,
    )
    human_strong = np.asarray(
        directional["HUMAN_NMDA"]["strong_norms_mV"],
        dtype=float,
    )

    weak_ratios = human_weak / np.maximum(frozen_weak, 1e-30)
    strong_ratios = human_strong / np.maximum(frozen_strong, 1e-30)
    weak_gain = float(np.median(weak_ratios))
    strong_gain = float(np.median(strong_ratios))
    selective_ratio = weak_gain / max(strong_gain, 1e-30)

    visible_gain = (
        metrics["HUMAN_NMDA"]["noise_visible_rank"]
        - metrics["FROZEN_NMDA"]["noise_visible_rank"]
    )
    accuracy_gain = (
        accuracy["HUMAN_NMDA"]["mean"]
        - accuracy["FROZEN_NMDA"]["mean"]
    )

    fd = finite_difference_check(
        graph,
        hidden_D,
        normalized_weights[:, selected[0]],
        "HUMAN_NMDA",
        matrices["HUMAN_NMDA"][0],
        parameter_index=0,
    )

    locked_requirements = {
        "passive_helper_reproduces_gate4": passive_reproduction_error <= 1e-10,
        "frozen_nmda_equals_rest_matched_ampa": (
            frozen_ampa_max_abs <= FROZEN_AMPA_EQUIVALENCE_ATOL_MV
        ),
        "human_implicit_derivative_fd_error_le_8pct": (
            fd["relative_error"] <= FD_MAX_RELATIVE_ERROR
        ),
        "human_visible_rank_gain_ge_1": (
            visible_gain >= RESCUE_MIN_VISIBLE_RANK_GAIN
        ),
        "human_accuracy_gain_ge_0p05": (
            accuracy_gain >= RESCUE_MIN_ACCURACY_GAIN
        ),
        "human_passive_weak_subspace_gain_ge_1p20": (
            weak_gain >= RESCUE_MIN_WEAK_GAIN
        ),
        "human_weak_gain_selective_ge_1p10": (
            selective_ratio >= RESCUE_MIN_WEAK_SELECTIVITY
        ),
    }

    controls_ok = all(
        locked_requirements[key]
        for key in (
            "passive_helper_reproduces_gate4",
            "frozen_nmda_equals_rest_matched_ampa",
            "human_implicit_derivative_fd_error_le_8pct",
        )
    )
    rescue = bool(
        controls_ok
        and locked_requirements["human_visible_rank_gain_ge_1"]
        and locked_requirements["human_accuracy_gain_ge_0p05"]
        and locked_requirements["human_passive_weak_subspace_gain_ge_1p20"]
        and locked_requirements["human_weak_gain_selective_ge_1p10"]
    )

    human_worse = bool(
        visible_gain < 0
        or accuracy_gain <= -0.02
        or weak_gain < 0.95
    )

    if rescue:
        classification = "HUMAN_NMDA_RESCUES_WEAK_SOMA_OBSERVABILITY"
    elif controls_ok and human_worse:
        classification = "HUMAN_NMDA_COMPRESSES_SOMA_OBSERVABILITY"
    elif controls_ok:
        classification = "HUMAN_NMDA_RESHAPES_BUT_DOES_NOT_RESCUE_SOMA_OBSERVABILITY"
    else:
        classification = "GATE5_CONTROL_FAILURE"

    result = {
        "gate": 5,
        "name": "nmda_observability_bridge",
        "classification": classification,
        "rescue_passed": rescue,
        "source": {
            "morphology_identifier": "1125",
            "model": "Gate-4 passive morphology graph plus released HUMAN static AMPA/NMDA law",
            "released_neuron_model": False,
        },
        "locked_protocol": {
            "probe_current_rest_matched_nA": PROBE_CURRENT_NA,
            "soma_noise_sigma_mV": NOISE_SIGMA_MV,
            "same_passive_selected_addresses_all_conditions": True,
            "probe_budget": PROBE_BUDGET,
            "hidden_parameters": HIDDEN_PARAMS,
            "human_gamma_per_mV": HUMAN_GAMMA_PER_MV,
            "raw_nmda_to_ampa": RAW_NMDA_TO_AMPA,
            "rescue_thresholds": {
                "visible_rank_gain_min": RESCUE_MIN_VISIBLE_RANK_GAIN,
                "accuracy_gain_min": RESCUE_MIN_ACCURACY_GAIN,
                "weak_gain_min": RESCUE_MIN_WEAK_GAIN,
                "weak_selectivity_min": RESCUE_MIN_WEAK_SELECTIVITY,
            },
        },
        "selected_probes": [
            {"probe_index": int(i), **probe_meta[i]}
            for i in selected
        ],
        "passive_singular_values_mV": s_passive.tolist(),
        "passive_weak_direction_count": int(weak_count),
        "metrics": metrics,
        "accuracy_at_1uv": accuracy,
        "directional": directional,
        "human_vs_frozen": {
            "visible_rank_gain": int(visible_gain),
            "accuracy_gain": float(accuracy_gain),
            "weak_direction_gain_ratios": weak_ratios.tolist(),
            "strong_direction_gain_ratios": strong_ratios.tolist(),
            "median_weak_gain": weak_gain,
            "median_strong_gain": strong_gain,
            "weak_selectivity_ratio": selective_ratio,
        },
        "controls": {
            "passive_reproduction_max_abs_mV": passive_reproduction_error,
            "frozen_vs_ampa_max_abs_mV": frozen_ampa_max_abs,
            "human_finite_difference_check": fd,
        },
        "run_meta": run_meta,
        "passive_helper_meta": passive_helper_meta,
        "locked_requirements": locked_requirements,
        "scope": (
            "Gate 5 tests whether the released HUMAN magnesium-block law changes "
            "the Gate-4 passive soma-observability spectrum on the real morphology. "
            "It does not instantiate the released FCI NEURON model, action potentials, "
            "plasticity, or autonomous dendritic self-interrogation."
        ),
    }

    (args.output_dir / "gate5.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )

    print("GeometricNeuronV24 Gate 5 - NMDA observability bridge")
    print()
    print("same 12 passive-selected addresses in every condition")
    print()
    for condition in (
        "PASSIVE_CURRENT",
        "AMPA_ONLY",
        "FROZEN_NMDA",
        "HUMAN_NMDA",
    ):
        m = metrics[condition]
        a = accuracy[condition]
        print(
            f"{condition:18s} "
            f"rank {m['numerical_rank']:2d}  "
            f"visible {m['noise_visible_rank']:2d}  "
            f"s_min {m['smallest_singular_mV']:.4e} mV  "
            f"acc {a['mean']:.3f}"
        )
    print()
    print(
        "HUMAN - frozen visible-rank gain:     "
        f"{visible_gain:+d}"
    )
    print(
        "HUMAN - frozen identity gain:         "
        f"{accuracy_gain:+.3f}"
    )
    print(
        "median passive-weak gain HUMAN/frozen:"
        f" {weak_gain:.3f}x"
    )
    print(
        "median passive-strong gain:           "
        f"{strong_gain:.3f}x"
    )
    print(
        "weak/strong selectivity:              "
        f"{selective_ratio:.3f}x"
    )
    print(
        "AMPA vs frozen max abs discrepancy:   "
        f"{frozen_ampa_max_abs:.3e} mV"
    )
    print(
        "HUMAN 10% finite-difference error:    "
        f"{fd['relative_error']:.4f}"
    )
    print()
    print(classification)

    # Control failure is a software/model failure. A scientifically negative
    # NMDA result is a valid Gate-5 completion and should stay green in CI.
    raise SystemExit(0 if controls_ok else 1)


if __name__ == "__main__":
    main()
