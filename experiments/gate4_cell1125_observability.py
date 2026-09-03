#!/usr/bin/env python3
"""Gate 4: soma-only observability on a real human dendritic morphology.

This is the first V24 experiment that puts the READ+WRITE question onto the
released human L2/3 cell-1125 morphology used by Operaattori.

The model is intentionally narrower than the released NEURON model.  We load
the pinned ASC morphology through MorphIO, build a passive morphology-graph
cable at DC, and hide one of several local leak-density changes.  A "probe"
is a current injection at a chosen dendritic address and spatial cluster scale;
the only readout is somatic voltage.

For the baseline passive matrix A, probe b gives x = A^-1 b.  A local fractional
leak perturbation D_p changes the soma response with analytic first-order
sensitivity

    dy/dp = -e_soma^T A^-1 D_p A^-1 b.

The left solve is a software adjoint.  Nothing here claims that a biological
backpropagating spike is an adjoint.

Primary question:
    Does choosing dendritic stimulation address increase the number and
    conditioning of hidden local parameter directions observable at the soma?

Secondary question:
    Does allowing multiple cluster scales add anything beyond point probes?

This gate does not use NMDA, spikes, plasticity, or a learned policy.  It earns
only a passive observability statement on one real released morphology.
"""

from __future__ import annotations

import argparse
import json
import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import splu


ROOT = Path(__file__).resolve().parents[1]

SOURCE_COMMIT = "75ad8b4d81a7f51bf888b30650c543592340db06"
SOURCE_NAME = "2013_03_06_cell11_1125_H41_06.asc"
SOURCE_REL = (
    "simulating_neurons/neuron_models/human/eyal/"
    "Human_L23_PC_0603_11_937_Eyal_passive_dends_simple_soma/"
    "morphologies/" + SOURCE_NAME
)
SOURCE_URL = (
    "https://raw.githubusercontent.com/ido4848/FCI/"
    + SOURCE_COMMIT + "/" + SOURCE_REL
)

RA_OHM_CM = 150.0
RM_OHM_CM2 = 20000.0
PROBE_CURRENT_NA = 0.10
HIDDEN_FRACTIONAL_LEAK_CHANGE = 0.10
HIDDEN_PARAMS = 12
PROBE_CENTERS = 48
CLUSTER_RADII_UM = (0.0, 35.0, 110.0)
PROBE_BUDGET = 12
RANDOM_AUDIT_DRAWS = 128
NOISE_SIGMA_MV = 0.001
DENDRITE_TYPES = (3, 4)


@dataclass(frozen=True)
class PointTree:
    positions: np.ndarray
    parents: np.ndarray
    radii: np.ndarray
    section_ids: np.ndarray
    section_types: np.ndarray
    soma_points: np.ndarray
    soma_radii: np.ndarray


def enum_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        raw = getattr(value, "value", None)
        return -1 if raw is None else int(raw)


def download_source(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 100_000:
        return dest
    req = urllib.request.Request(
        SOURCE_URL,
        headers={"User-Agent": "GeometricNeuronV24-Gate4/1.0"},
    )
    with urllib.request.urlopen(req, timeout=60) as response, dest.open("wb") as f:
        f.write(response.read())
    if dest.stat().st_size < 100_000:
        raise RuntimeError("downloaded morphology is unexpectedly small")
    return dest


def load_morphio_tree(path: str | Path, duplicate_tol: float = 1e-5) -> PointTree:
    from morphio import Morphology

    morph = Morphology(str(path))
    soma_points = np.asarray(morph.soma.points, dtype=float)
    soma_diameters = np.asarray(morph.soma.diameters, dtype=float)

    if len(soma_points):
        root = np.mean(soma_points, axis=0)
        root_radius = (
            float(np.mean(soma_diameters) / 2.0)
            if len(soma_diameters)
            else 1.0
        )
    else:
        first = [
            np.asarray(section.points[0], dtype=float)
            for section in morph.root_sections
            if len(section.points)
        ]
        if not first:
            raise ValueError("morphology has no soma and no neurite points")
        root = np.mean(np.stack(first), axis=0)
        root_radius = 1.0

    positions: list[np.ndarray] = [root]
    parents: list[int] = [-1]
    radii: list[float] = [root_radius]
    section_ids: list[int] = [-1]
    section_types: list[int] = [1]
    section_end: dict[int, int] = {}

    for section in morph.iter():
        sid = int(section.id)
        stype = enum_int(section.type)
        if section.is_root:
            current_parent = 0
        else:
            pid = int(section.parent.id)
            if pid not in section_end:
                raise ValueError(f"parent section {pid} missing before child {sid}")
            current_parent = section_end[pid]

        pts = np.asarray(section.points, dtype=float)
        diams = np.asarray(section.diameters, dtype=float)
        if len(pts) != len(diams):
            raise ValueError(f"section {sid}: point/diameter mismatch")
        if not len(pts):
            section_end[sid] = current_parent
            continue

        start = 0
        if np.linalg.norm(pts[0] - positions[current_parent]) <= duplicate_tol:
            start = 1

        for j in range(start, len(pts)):
            p = np.asarray(pts[j], dtype=float)
            if np.linalg.norm(p - positions[current_parent]) <= duplicate_tol:
                continue
            idx = len(positions)
            positions.append(p)
            parents.append(current_parent)
            radii.append(float(diams[j]) / 2.0)
            section_ids.append(sid)
            section_types.append(stype)
            current_parent = idx

        section_end[sid] = current_parent

    tree = PointTree(
        positions=np.asarray(positions, dtype=float),
        parents=np.asarray(parents, dtype=np.int64),
        radii=np.asarray(radii, dtype=float),
        section_ids=np.asarray(section_ids, dtype=np.int64),
        section_types=np.asarray(section_types, dtype=np.int64),
        soma_points=np.asarray(soma_points, dtype=float).reshape((-1, 3)),
        soma_radii=np.asarray(soma_diameters, dtype=float) / 2.0,
    )
    if len(tree.positions) < 1000:
        raise RuntimeError("parsed morphology is unexpectedly small")
    return tree


def axial_conductance_uS(length_um: float, radius_um: float) -> float:
    length_cm = max(float(length_um), 1e-9) * 1e-4
    radius_cm = max(float(radius_um), 0.05) * 1e-4
    area_cm2 = math.pi * radius_cm * radius_cm
    resistance_ohm = RA_OHM_CM * length_cm / area_cm2
    return 1e6 / resistance_ohm


def frustum_area_um2(length_um: float, r0_um: float, r1_um: float) -> float:
    slant = math.sqrt(max(length_um, 0.0) ** 2 + (r1_um - r0_um) ** 2)
    return math.pi * max(r0_um + r1_um, 0.1) * slant


def cable_graph(tree: PointTree, *, flatten_radii: bool = False) -> dict:
    n = len(tree.positions)
    radii = tree.radii.copy()
    dend_mask = np.isin(tree.section_types, DENDRITE_TYPES)
    if flatten_radii:
        median = float(np.median(radii[dend_mask & (radii > 0)]))
        radii[dend_mask] = median

    membrane_area = np.zeros(n, dtype=float)
    soma_r = (
        float(np.mean(tree.soma_radii[tree.soma_radii > 0]))
        if np.any(tree.soma_radii > 0)
        else max(float(radii[0]), 1.0)
    )
    membrane_area[0] += 4.0 * math.pi * soma_r * soma_r

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    lengths = np.zeros(n, dtype=float)

    for i in range(1, n):
        p = int(tree.parents[i])
        length = float(np.linalg.norm(tree.positions[i] - tree.positions[p]))
        lengths[i] = length
        r0 = float(radii[p])
        r1 = float(radii[i])
        area = frustum_area_um2(length, r0, r1)
        membrane_area[p] += 0.5 * area
        membrane_area[i] += 0.5 * area

        # The synthetic soma-centroid attachment is not a cylindrical neurite.
        # Treat it as a short soma-to-neurite access rather than its literal
        # centroid-to-contour distance.
        if p == 0:
            effective_length = min(max(length, 0.5), max(2.0 * soma_r, 1.0))
            effective_radius = max(r1, min(soma_r, 4.0 * r1))
        else:
            effective_length = max(length, 1e-6)
            effective_radius = max(0.5 * (r0 + r1), 0.05)

        g = axial_conductance_uS(effective_length, effective_radius)
        rows.extend([i, p, i, p])
        cols.extend([i, p, p, i])
        vals.extend([g, g, -g, -g])

    leak_uS = membrane_area * 1e-8 / RM_OHM_CM2 * 1e6
    leak_uS = np.maximum(leak_uS, 1e-12)

    rows.extend(range(n))
    cols.extend(range(n))
    vals.extend(float(x) for x in leak_uS)

    A = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsc()
    A.sum_duplicates()

    path_um = np.zeros(n, dtype=float)
    for i in range(1, n):
        path_um[i] = path_um[int(tree.parents[i])] + lengths[i]

    return {
        "A_uS": A,
        "leak_uS": leak_uS,
        "membrane_area_um2": membrane_area,
        "edge_length_um": lengths,
        "path_um": path_um,
        "radii_um": radii,
    }


def section_metadata(tree: PointTree, graph: dict) -> list[dict]:
    sections: dict[int, list[int]] = {}
    for i, sid in enumerate(tree.section_ids):
        sid = int(sid)
        if sid < 0:
            continue
        sections.setdefault(sid, []).append(i)

    out = []
    for sid, nodes in sections.items():
        arr = np.asarray(nodes, dtype=int)
        stypes = tree.section_types[arr]
        stype = int(np.bincount(np.maximum(stypes, 0)).argmax())
        if stype not in DENDRITE_TYPES:
            continue
        length = float(np.sum(graph["edge_length_um"][arr]))
        area = float(np.sum(graph["membrane_area_um2"][arr]))
        path = float(np.mean(graph["path_um"][arr]))
        if len(arr) < 4 or length < 18.0 or area <= 0.0:
            continue
        midpoint = int(arr[np.argmin(np.abs(graph["path_um"][arr] - path))])
        out.append(
            {
                "section_id": sid,
                "section_type": stype,
                "nodes": arr,
                "node_count": int(len(arr)),
                "length_um": length,
                "area_um2": area,
                "mean_path_um": path,
                "midpoint_node": midpoint,
            }
        )
    out.sort(key=lambda row: (row["mean_path_um"], row["section_id"]))
    return out


def quantile_pick(rows: list[dict], count: int) -> list[dict]:
    if len(rows) < count:
        raise RuntimeError(f"need {count} sections, found {len(rows)}")
    positions = np.linspace(0, len(rows) - 1, count)
    picked = []
    used = set()
    for value in positions:
        idx = int(round(float(value)))
        while idx in used and idx + 1 < len(rows):
            idx += 1
        while idx in used and idx - 1 >= 0:
            idx -= 1
        if idx in used:
            raise RuntimeError("could not choose unique section quantiles")
        used.add(idx)
        picked.append(rows[idx])
    return picked


def choose_hidden_sections(rows: list[dict]) -> list[dict]:
    # Remove the most proximal and most extreme distal 5% so hidden regions are
    # not trivially all soma-adjacent or terminal tips.
    lo = int(round(0.05 * len(rows)))
    hi = max(lo + HIDDEN_PARAMS, int(round(0.95 * len(rows))))
    trimmed = rows[lo:hi]
    return quantile_pick(trimmed, HIDDEN_PARAMS)


def choose_probe_centers(rows: list[dict]) -> list[dict]:
    return quantile_pick(rows, PROBE_CENTERS)


def tree_distance_matrix(tree: PointTree, graph: dict, centers: list[int]) -> np.ndarray:
    n = len(tree.positions)
    lengths = graph["edge_length_um"]
    rows = []
    cols = []
    vals = []
    for i in range(1, n):
        p = int(tree.parents[i])
        w = max(float(lengths[i]), 1e-9)
        rows.extend([i, p])
        cols.extend([p, i])
        vals.extend([w, w])
    adjacency = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    return np.asarray(dijkstra(adjacency, directed=False, indices=np.asarray(centers, dtype=int)))


def build_probes(
    tree: PointTree,
    graph: dict,
    centers: list[dict],
) -> tuple[np.ndarray, list[dict]]:
    center_nodes = [int(row["midpoint_node"]) for row in centers]
    distances = tree_distance_matrix(tree, graph, center_nodes)
    dend_mask = np.isin(tree.section_types, DENDRITE_TYPES)
    area = graph["membrane_area_um2"]

    probes = []
    meta: list[dict] = []
    for ci, center in enumerate(centers):
        node = int(center["midpoint_node"])
        for radius in CLUSTER_RADII_UM:
            weights = np.zeros(len(tree.positions), dtype=float)
            if radius <= 0.0:
                weights[node] = 1.0
                support = np.asarray([node], dtype=int)
            else:
                mask = (distances[ci] <= float(radius)) & dend_mask
                support = np.flatnonzero(mask)
                if len(support) == 0:
                    support = np.asarray([node], dtype=int)
                local_area = np.maximum(area[support], 1e-12)
                weights[support] = local_area / np.sum(local_area)
            probes.append(PROBE_CURRENT_NA * weights)
            meta.append(
                {
                    "center_index": ci,
                    "center_node": node,
                    "center_section_id": int(center["section_id"]),
                    "center_path_um": float(center["mean_path_um"]),
                    "radius_um": float(radius),
                    "support_nodes": int(len(support)),
                }
            )
    return np.stack(probes, axis=1), meta


def hidden_derivative_matrix(tree: PointTree, graph: dict, hidden: list[dict]) -> np.ndarray:
    n = len(tree.positions)
    D = np.zeros((n, len(hidden)), dtype=float)
    leak = graph["leak_uS"]
    for j, section in enumerate(hidden):
        nodes = np.asarray(section["nodes"], dtype=int)
        D[nodes, j] = HIDDEN_FRACTIONAL_LEAK_CHANGE * leak[nodes]
    return D


def sensitivity_matrix(
    tree: PointTree,
    graph: dict,
    hidden: list[dict],
    probe_matrix: np.ndarray,
) -> tuple[np.ndarray, dict]:
    A = graph["A_uS"]
    lu = splu(A)
    soma = np.zeros(len(tree.positions), dtype=float)
    soma[0] = 1.0
    adjoint = lu.solve(soma, trans="T")
    X = lu.solve(np.asarray(probe_matrix, dtype=float))
    D = hidden_derivative_matrix(tree, graph, hidden)
    weighted = adjoint[:, None] * D
    J = -(X.T @ weighted)
    soma_baseline = X[0, :]
    return J, {
        "lu": lu,
        "adjoint": adjoint,
        "state": X,
        "soma_baseline_mV": soma_baseline,
    }


def singular_metrics(J: np.ndarray) -> dict:
    s = np.linalg.svd(np.asarray(J, dtype=float), compute_uv=False)
    if len(s) == 0:
        return {
            "singular_values_mV": [],
            "numerical_rank": 0,
            "noise_visible_rank": 0,
            "effective_rank": 0.0,
            "smallest_singular_mV": 0.0,
            "logdet_fisher": float("-inf"),
        }
    tol = max(float(s[0]) * 1e-8, 1e-15)
    rank = int(np.count_nonzero(s > tol))
    visible = int(np.count_nonzero(s >= NOISE_SIGMA_MV))
    energy = s * s
    eff = float((np.sum(energy) ** 2) / (np.sum(energy * energy) + 1e-30))
    fisher_eigs = (s / NOISE_SIGMA_MV) ** 2
    logdet = float(np.sum(np.log1p(fisher_eigs)))
    return {
        "singular_values_mV": s.tolist(),
        "numerical_rank": rank,
        "noise_visible_rank": visible,
        "effective_rank": eff,
        "smallest_singular_mV": float(s[-1]),
        "largest_singular_mV": float(s[0]),
        "condition_number": float(s[0] / max(s[-1], 1e-30)),
        "logdet_fisher": logdet,
    }


def greedy_logdet(J: np.ndarray, candidates: np.ndarray, budget: int) -> list[int]:
    candidates = np.asarray(candidates, dtype=int)
    p = J.shape[1]
    F = np.eye(p, dtype=float) * 1e-8
    chosen: list[int] = []
    available = set(int(i) for i in candidates)
    for _ in range(min(int(budget), len(available))):
        best = None
        best_score = -np.inf
        for i in available:
            row = J[i] / NOISE_SIGMA_MV
            trial = F + np.outer(row, row)
            sign, logdet = np.linalg.slogdet(trial)
            score = float(logdet) if sign > 0 else -np.inf
            if score > best_score:
                best_score = score
                best = int(i)
        if best is None:
            break
        row = J[best] / NOISE_SIGMA_MV
        F += np.outer(row, row)
        chosen.append(best)
        available.remove(best)
    return chosen


def fixed_site_indices(meta: list[dict], budget: int) -> list[int]:
    center_index = len({row["center_index"] for row in meta}) // 2
    local = [i for i, row in enumerate(meta) if row["center_index"] == center_index]
    if not local:
        local = [0]
    out = []
    while len(out) < budget:
        out.extend(local)
    return out[:budget]


def random_audit(
    J: np.ndarray,
    candidates: np.ndarray,
    budget: int,
    seed: int,
) -> dict:
    candidates = np.asarray(candidates, dtype=int)
    rng = np.random.default_rng(seed)
    metrics = []
    for _ in range(RANDOM_AUDIT_DRAWS):
        selected = rng.choice(candidates, size=min(budget, len(candidates)), replace=False)
        metrics.append(singular_metrics(J[selected]))
    smallest = np.asarray([m["smallest_singular_mV"] for m in metrics], dtype=float)
    ranks = np.asarray([m["numerical_rank"] for m in metrics], dtype=float)
    visible = np.asarray([m["noise_visible_rank"] for m in metrics], dtype=float)
    logdet = np.asarray([m["logdet_fisher"] for m in metrics], dtype=float)
    return {
        "draws": RANDOM_AUDIT_DRAWS,
        "smallest_singular_mV": {
            "median": float(np.median(smallest)),
            "p10": float(np.quantile(smallest, 0.10)),
            "p90": float(np.quantile(smallest, 0.90)),
            "max": float(np.max(smallest)),
        },
        "numerical_rank": {
            "median": float(np.median(ranks)),
            "min": int(np.min(ranks)),
            "max": int(np.max(ranks)),
        },
        "noise_visible_rank": {
            "median": float(np.median(visible)),
            "min": int(np.min(visible)),
            "max": int(np.max(visible)),
        },
        "logdet_fisher": {
            "median": float(np.median(logdet)),
            "p10": float(np.quantile(logdet, 0.10)),
            "p90": float(np.quantile(logdet, 0.90)),
        },
    }


def fingerprint_accuracy(
    J: np.ndarray,
    selected: list[int],
    *,
    seed: int,
    trials: int = 4096,
) -> float:
    # One of P hidden 10%-leak perturbations occurs.  The linearized response
    # fingerprint is the corresponding J column.  Add independent soma noise
    # to each probe and classify by nearest template.
    M = np.asarray(J[selected], dtype=float)
    templates = M.T
    rng = np.random.default_rng(seed)
    truth = rng.integers(0, templates.shape[0], size=trials)
    observed = templates[truth] + rng.normal(
        scale=NOISE_SIGMA_MV,
        size=(trials, templates.shape[1]),
    )
    d2 = np.sum((observed[:, None, :] - templates[None, :, :]) ** 2, axis=2)
    pred = np.argmin(d2, axis=1)
    return float(np.mean(pred == truth))


def finite_difference_check(
    tree: PointTree,
    graph: dict,
    hidden: list[dict],
    probe_matrix: np.ndarray,
    J: np.ndarray,
    selected_probe: int,
    selected_param: int,
) -> dict:
    A = graph["A_uS"].copy().tolil()
    nodes = np.asarray(hidden[selected_param]["nodes"], dtype=int)
    leak = graph["leak_uS"]
    for node in nodes:
        A[node, node] += HIDDEN_FRACTIONAL_LEAK_CHANGE * leak[node]
    lu = splu(A.tocsc())
    x = lu.solve(probe_matrix[:, selected_probe])
    perturbed = float(x[0])

    base_lu = splu(graph["A_uS"])
    x0 = base_lu.solve(probe_matrix[:, selected_probe])
    baseline = float(x0[0])
    exact_delta = perturbed - baseline
    linear_delta = float(J[selected_probe, selected_param])
    rel = abs(exact_delta - linear_delta) / (abs(exact_delta) + 1e-30)
    return {
        "probe_index": int(selected_probe),
        "parameter_index": int(selected_param),
        "baseline_soma_mV": baseline,
        "perturbed_soma_mV": perturbed,
        "exact_delta_mV": exact_delta,
        "linearized_delta_mV": linear_delta,
        "relative_error": float(rel),
    }


def run(seed: int, output_dir: Path, morphology: Path | None = None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    source = morphology or download_source(output_dir / "_source" / SOURCE_NAME)
    tree = load_morphio_tree(source)
    graph = cable_graph(tree)
    sections = section_metadata(tree, graph)
    hidden = choose_hidden_sections(sections)
    centers = choose_probe_centers(sections)
    probes, meta = build_probes(tree, graph, centers)
    J, solved = sensitivity_matrix(tree, graph, hidden, probes)

    all_indices = np.arange(len(meta), dtype=int)
    point_indices = np.asarray(
        [i for i, row in enumerate(meta) if row["radius_um"] == 0.0],
        dtype=int,
    )

    active_all = greedy_logdet(J, all_indices, PROBE_BUDGET)
    active_point = greedy_logdet(J, point_indices, PROBE_BUDGET)
    fixed = fixed_site_indices(meta, PROBE_BUDGET)

    active_all_metrics = singular_metrics(J[active_all])
    active_point_metrics = singular_metrics(J[active_point])
    fixed_metrics = singular_metrics(J[fixed])
    random_all = random_audit(J, all_indices, PROBE_BUDGET, seed + 100)
    random_point = random_audit(J, point_indices, PROBE_BUDGET, seed + 200)

    active_accuracy = fingerprint_accuracy(J, active_all, seed=seed + 300)
    active_point_accuracy = fingerprint_accuracy(J, active_point, seed=seed + 301)
    fixed_accuracy = fingerprint_accuracy(J, fixed, seed=seed + 302)

    # One exact nonlinear-in-parameter solve validates the local derivative.
    fd = finite_difference_check(
        tree,
        graph,
        hidden,
        probes,
        J,
        selected_probe=int(active_all[0]),
        selected_param=0,
    )

    chosen_scales = {
        str(radius): int(
            sum(abs(meta[i]["radius_um"] - radius) < 1e-12 for i in active_all)
        )
        for radius in CLUSTER_RADII_UM
    }

    random_median_smin = random_all["smallest_singular_mV"]["median"]
    requirements = {
        "real_morphology_has_gt_1000_nodes": len(tree.positions) > 1000,
        "active_reaches_full_numerical_rank": (
            active_all_metrics["numerical_rank"] == HIDDEN_PARAMS
        ),
        "active_smallest_singular_beats_random_median_by_20pct": (
            active_all_metrics["smallest_singular_mV"]
            >= 1.20 * random_median_smin
        ),
        "active_fingerprint_accuracy_ge_0_90": active_accuracy >= 0.90,
        "fixed_address_rank_le_3": fixed_metrics["numerical_rank"] <= len(CLUSTER_RADII_UM),
        "analytic_sensitivity_matches_10pct_finite_difference_within_8pct": (
            fd["relative_error"] <= 0.08
        ),
    }
    passed = all(requirements.values())

    # Radius-flattened morphology is a secondary attacker only; there is no
    # positive/negative threshold because the direction was not assumed.
    flat_graph = cable_graph(tree, flatten_radii=True)
    J_flat, _ = sensitivity_matrix(tree, flat_graph, hidden, probes)
    flat_active = greedy_logdet(J_flat, all_indices, PROBE_BUDGET)
    flat_metrics = singular_metrics(J_flat[flat_active])

    result = {
        "gate": 4,
        "name": "cell1125_soma_observability",
        "seed": seed,
        "passed": passed,
        "classification": (
            "ADDRESSED_DENDRITIC_PERTURBATIONS_OPEN_SOMA_OBSERVABILITY"
            if passed
            else "CELL1125_OBSERVABILITY_GATE_FAILED"
        ),
        "source": {
            "repository": "ido4848/FCI",
            "commit": SOURCE_COMMIT,
            "path": SOURCE_REL,
            "morphology_identifier": "1125",
            "cell": "human L2/3 pyramidal neuron",
        },
        "model": {
            "kind": "MorphIO point-tree passive DC cable",
            "Ra_ohm_cm": RA_OHM_CM,
            "Rm_ohm_cm2": RM_OHM_CM2,
            "probe_current_nA": PROBE_CURRENT_NA,
            "hidden_change": "10% local leak-density increase on one selected dendritic section",
            "soma_readout_only": True,
            "software_adjoint": True,
            "uses_nmda": False,
            "uses_spikes": False,
            "uses_neuron_simulator": False,
        },
        "morphology": {
            "nodes": int(len(tree.positions)),
            "dendritic_sections_eligible": int(len(sections)),
            "bbox_um": (
                np.max(tree.positions, axis=0) - np.min(tree.positions, axis=0)
            ).tolist(),
            "total_membrane_area_um2": float(np.sum(graph["membrane_area_um2"])),
        },
        "hidden_parameters": [
            {
                "parameter_index": j,
                "section_id": int(row["section_id"]),
                "section_type": int(row["section_type"]),
                "node_count": int(row["node_count"]),
                "length_um": float(row["length_um"]),
                "area_um2": float(row["area_um2"]),
                "mean_path_um": float(row["mean_path_um"]),
            }
            for j, row in enumerate(hidden)
        ],
        "probe_family": {
            "centers": PROBE_CENTERS,
            "cluster_radii_um": list(CLUSTER_RADII_UM),
            "candidate_probes": int(len(meta)),
            "budget": PROBE_BUDGET,
            "noise_sigma_mV": NOISE_SIGMA_MV,
            "total_current_matched_across_scales": True,
        },
        "policies": {
            "fixed_one_address_vary_scale": {
                "selected": [meta[i] for i in fixed],
                "metrics": fixed_metrics,
                "fingerprint_accuracy": fixed_accuracy,
            },
            "active_point_only": {
                "selected": [meta[i] for i in active_point],
                "metrics": active_point_metrics,
                "fingerprint_accuracy": active_point_accuracy,
            },
            "active_multiscale": {
                "selected": [meta[i] for i in active_all],
                "selected_scale_counts": chosen_scales,
                "metrics": active_all_metrics,
                "fingerprint_accuracy": active_accuracy,
            },
            "random_point_only": random_point,
            "random_multiscale": random_all,
        },
        "finite_difference_check": fd,
        "radius_flattened_attacker": {
            "description": "all dendritic radii replaced by the real-cell median before rebuilding the passive cable",
            "active_metrics": flat_metrics,
            "selected": [meta[i] for i in flat_active],
        },
        "locked_requirements": requirements,
        "scope": (
            "The morphology is real and pinned, but the electrical model is a "
            "passive point-tree DC cable built directly from morphology. Hidden "
            "changes are a known finite set of local leak perturbations. This is "
            "not the released FCI NEURON model and not evidence of autonomous "
            "self-interrogation by a neuron."
        ),
    }

    (output_dir / "gate4.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def print_receipt(result: dict) -> None:
    p = result["policies"]
    print("GeometricNeuronV24 Gate 4 - cell 1125 soma observability")
    print()
    print(f"morphology nodes:                     {result['morphology']['nodes']}")
    print(f"eligible dendritic sections:          {result['morphology']['dendritic_sections_eligible']}")
    print(f"hidden local parameters:              {len(result['hidden_parameters'])}")
    print(f"candidate probes / budget:            {result['probe_family']['candidate_probes']} / {result['probe_family']['budget']}")
    print()
    for name in ("fixed_one_address_vary_scale", "active_point_only", "active_multiscale"):
        row = p[name]
        m = row["metrics"]
        print(
            f"{name:34s} rank {m['numerical_rank']:2d}  "
            f"s_min {m['smallest_singular_mV']:.4e} mV  "
            f"visible {m['noise_visible_rank']:2d}  "
            f"acc {row['fingerprint_accuracy']:.3f}"
        )
    r = p["random_multiscale"]
    print(
        f"{'random_multiscale median':34s} rank {r['numerical_rank']['median']:.1f}  "
        f"s_min {r['smallest_singular_mV']['median']:.4e} mV  "
        f"visible {r['noise_visible_rank']['median']:.1f}"
    )
    print(
        "active multiscale scale counts:       "
        f"{p['active_multiscale']['selected_scale_counts']}"
    )
    print(
        "10% sensitivity finite-diff rel err: "
        f"{result['finite_difference_check']['relative_error']:.4f}"
    )
    print()
    print(result["classification"])
    print(
        "This earns a passive soma-observability result on the pinned real "
        "morphology only. It does not yet test NMDA, spikes, biological active "
        "sensing, or a physical adjoint."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=24031976)
    ap.add_argument("--morphology", type=Path)
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "gate4",
    )
    args = ap.parse_args()
    result = run(args.seed, args.output_dir, args.morphology)
    print_receipt(result)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
