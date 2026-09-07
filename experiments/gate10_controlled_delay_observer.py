#!/usr/bin/env python3
"""Gate 10: controlled delay reconstruction of a genuinely moving hidden state.

Gates 7-9R showed that short histories of (probe address, returned scalar) can
replace perfect bookkeeping in a *stationary* inverse problem.  That is useful,
but it is not yet the setting where Takens-style state reconstruction earns its
name.

Gate 10 introduces a moving hidden state and a known self-action.

Hidden state per episode:

    theta_t      phase on a circle
    omega        unknown constant angular drift

Before observation t the system issues one of four meaningless actions a_t.
That action has two known physical consequences:

    1. it applies a small phase kick kappa[a_t]
    2. it selects a scalar sensor direction phi[a_t]

Then

    theta_t = theta_(t-1) + omega + kappa[a_t]
    y_t     = cos(theta_t - phi[a_t]) + noise

The observer never sees theta or omega.  It sees only the recent history of
(action, scalar return).  For a candidate hidden state (theta_t, omega), it
replays the known actions backward through the delay window and asks which
candidate best explains the returns.  The chosen candidate predicts the next
return *given the next outgoing action/efference copy*.

This is a small, explicit forced/input-output delay reconstruction.  It is not a
proof/application of classical autonomous Takens theorem; the action labels are
part of the observation history on purpose.

Attackers:
- K=1..8 history length.
- wrong-history-address: keep scalar returns but rotate every past action label.
- no-next-efference: reconstruct state correctly but do not know which next
  action will be issued; predict the action-average return.

A second test inserts occasional unreported external phase jumps.  The key
metric is whether |actual - predicted self-return| detects those hidden shocks
better than raw |y_t - y_(t-1)|.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Config:
    n_actions: int = 4
    steps: int = 700
    burn: int = 20
    seeds: int = 8
    obs_noise: float = 0.015
    omega_lo: float = 0.07
    omega_hi: float = 0.21
    theta_grid: int = 144
    omega_grid: int = 49
    shock_probability: float = 0.055
    shock_magnitude: float = 1.15


CFG = Config()
PHI = np.asarray([0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi], dtype=float)
KAPPA = np.asarray([-0.085, -0.025, 0.035, 0.095], dtype=float)
WINDOWS = (1, 2, 3, 4, 6, 8)


def simulate(seed: int, *, shocks: bool) -> dict:
    rng = np.random.default_rng(10_700_001 + int(seed) * 1009 + int(shocks))
    omega = float(rng.uniform(CFG.omega_lo, CFG.omega_hi))
    theta = float(rng.uniform(-np.pi, np.pi))
    actions = rng.integers(0, CFG.n_actions, size=CFG.steps, dtype=np.int64)
    y = np.zeros(CFG.steps, dtype=float)
    shock = np.zeros(CFG.steps, dtype=np.int64)
    theta_true = np.zeros(CFG.steps, dtype=float)

    for t, action in enumerate(actions):
        theta = theta + omega + float(KAPPA[action])
        if shocks and t >= CFG.burn and rng.random() < CFG.shock_probability:
            theta += float(rng.choice([-1.0, 1.0])) * CFG.shock_magnitude
            shock[t] = 1
        theta = float(np.arctan2(np.sin(theta), np.cos(theta)))
        theta_true[t] = theta
        y[t] = np.cos(theta - PHI[action]) + float(
            rng.normal(scale=CFG.obs_noise)
        )

    return {
        "actions": actions,
        "y": y,
        "shock": shock,
        "theta": theta_true,
        "omega": omega,
    }


THETA_CAND = np.linspace(-np.pi, np.pi, CFG.theta_grid, endpoint=False)
OMEGA_CAND = np.linspace(CFG.omega_lo, CFG.omega_hi, CFG.omega_grid)
THETA_MESH, OMEGA_MESH = np.meshgrid(THETA_CAND, OMEGA_CAND, indexing="ij")
THETA_FLAT = THETA_MESH.reshape(-1)
OMEGA_FLAT = OMEGA_MESH.reshape(-1)


def reconstruct_current(
    actions: np.ndarray,
    y: np.ndarray,
    t: int,
    k: int,
    *,
    wrong_history_address: bool = False,
) -> tuple[float, float, float]:
    """Grid-MAP estimate of current theta and omega from last k pairs.

    Returns (theta_hat, omega_hat, mean_squared_fit_error).
    """
    start = max(0, t - k + 1)
    idxs = np.arange(start, t + 1)
    a = np.asarray(actions[idxs], dtype=np.int64).copy()
    yy = np.asarray(y[idxs], dtype=float)
    if wrong_history_address:
        a = (a + 1) % CFG.n_actions

    # Candidate current state is theta_t.  Replay known dynamics backward.
    cost = np.zeros_like(THETA_FLAT)
    # cumulative kicks between past time j and current time t
    for local, j in enumerate(idxs):
        n_back = int(t - j)
        if n_back == 0:
            theta_j = THETA_FLAT
        else:
            # theta_t = theta_j + n_back*omega + sum kicks[j+1:t]
            future_actions = a[local + 1 :]
            kick_sum = float(np.sum(KAPPA[future_actions]))
            theta_j = THETA_FLAT - n_back * OMEGA_FLAT - kick_sum
        pred = np.cos(theta_j - PHI[a[local]])
        cost += (pred - yy[local]) ** 2

    best = int(np.argmin(cost))
    return (
        float(THETA_FLAT[best]),
        float(OMEGA_FLAT[best]),
        float(cost[best] / max(len(idxs), 1)),
    )


def predict_next(
    theta_hat: float,
    omega_hat: float,
    next_action: int,
    *,
    know_next_action: bool,
) -> float:
    if know_next_action:
        theta_next = theta_hat + omega_hat + float(KAPPA[next_action])
        return float(np.cos(theta_next - PHI[next_action]))

    # No efference copy of which action is about to be issued.  Best generic
    # prediction under the experiment's uniform random action policy.
    vals = []
    for action in range(CFG.n_actions):
        theta_next = theta_hat + omega_hat + float(KAPPA[action])
        vals.append(np.cos(theta_next - PHI[action]))
    return float(np.mean(vals))


def prediction_run(
    traj: dict,
    k: int,
    *,
    wrong_history_address: bool = False,
    know_next_action: bool = True,
) -> dict:
    actions = traj["actions"]
    y = traj["y"]
    shock = traj["shock"]
    preds = []
    actual = []
    labels = []
    raw_changes = []
    fit_errors = []
    omega_errors = []

    for t in range(max(CFG.burn, k - 1), len(y) - 1):
        theta_hat, omega_hat, fit = reconstruct_current(
            actions,
            y,
            t,
            k,
            wrong_history_address=wrong_history_address,
        )
        pred = predict_next(
            theta_hat,
            omega_hat,
            int(actions[t + 1]),
            know_next_action=know_next_action,
        )
        preds.append(pred)
        actual.append(float(y[t + 1]))
        labels.append(int(shock[t + 1]))
        raw_changes.append(abs(float(y[t + 1] - y[t])))
        fit_errors.append(fit)
        omega_errors.append(abs(float(omega_hat - traj["omega"])))

    preds = np.asarray(preds)
    actual = np.asarray(actual)
    labels = np.asarray(labels, dtype=np.int64)
    innovation = np.abs(actual - preds)
    return {
        "mse": float(np.mean((actual - preds) ** 2)),
        "mae": float(np.mean(innovation)),
        "fit_error": float(np.mean(fit_errors)),
        "omega_mae": float(np.mean(omega_errors)),
        "labels": labels,
        "innovation": innovation,
        "raw_change": np.asarray(raw_changes),
    }


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = np.asarray(scores[labels == 1], dtype=float)
    neg = np.asarray(scores[labels == 0], dtype=float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    total = 0.0
    for s in pos:
        total += float(np.sum(s > neg)) + 0.5 * float(np.sum(s == neg))
    return float(total / (len(pos) * len(neg)))


def aggregate(runs: list[dict]) -> dict:
    return {
        "mse": float(np.mean([r["mse"] for r in runs])),
        "mae": float(np.mean([r["mae"] for r in runs])),
        "fit_error": float(np.mean([r["fit_error"] for r in runs])),
        "omega_mae": float(np.mean([r["omega_mae"] for r in runs])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results" / "gate10"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    clean = [simulate(seed, shocks=False) for seed in range(CFG.seeds)]
    shocked = [simulate(100 + seed, shocks=True) for seed in range(CFG.seeds)]

    sweep = []
    for k in WINDOWS:
        sweep.append({
            "window": int(k),
            **aggregate([prediction_run(traj, k) for traj in clean]),
        })

    best = min(sweep, key=lambda row: row["mse"])
    best_k = int(best["window"])

    wrong = aggregate([
        prediction_run(traj, best_k, wrong_history_address=True)
        for traj in clean
    ])
    no_next = aggregate([
        prediction_run(traj, best_k, know_next_action=False)
        for traj in clean
    ])

    shock_runs = [prediction_run(traj, best_k) for traj in shocked]
    labels = np.concatenate([r["labels"] for r in shock_runs])
    innovation = np.concatenate([r["innovation"] for r in shock_runs])
    raw = np.concatenate([r["raw_change"] for r in shock_runs])

    shock_wrong_runs = [
        prediction_run(traj, best_k, wrong_history_address=True)
        for traj in shocked
    ]
    shock_wrong = np.concatenate([r["innovation"] for r in shock_wrong_runs])

    summary = {
        "schema": "geometric-neuron/gate10-controlled-delay-observer-v1",
        "question": (
            "Can finite history of (known self-action, scalar return) reconstruct a moving hidden state well enough to predict self-generated sensory return and expose unreported external change as innovation?"
        ),
        "dynamics": {
            "hidden_state": "theta plus unknown per-episode angular drift omega",
            "action_effect": "known phase kick plus action-specific scalar sensor direction",
            "actions": CFG.n_actions,
            "observation_noise": CFG.obs_noise,
        },
        "clean_prediction_by_history": sweep,
        "best_window": best_k,
        "attackers_at_best_window": {
            "wrong_history_address": wrong,
            "no_next_efference_copy": no_next,
        },
        "hidden_phase_shock": {
            "n_shocks": int(labels.sum()),
            "n_clean": int((labels == 0).sum()),
            "auc_raw_abs_change": auc(raw, labels),
            "auc_abs_innovation": auc(innovation, labels),
            "auc_abs_innovation_wrong_history_address": auc(shock_wrong, labels),
        },
        "claim_boundary": (
            "This is an explicit grid-based forced/input-output delay reconstruction on a synthetic controlled oscillator. It demonstrates or falsifies a computational composition of delay history and efference information; it is not a proof of Takens theorem in neurons, a cerebellar model, a dendritic cable implementation, or an AIS-grating mechanism."
        ),
    }

    path = args.output_dir / "gate10_summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
