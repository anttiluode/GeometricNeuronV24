# Results ledger

All headline values below come from deterministic seed `24031976`. Full arrays and thresholds are in the JSON receipts.

## Gate 0 - `LENS_ALGEBRA_EARNED_RANDOM_IMAGE_NULL_CONFIRMED`

| measurement | result |
|---|---:|
| hidden dimensions | 1024 |
| dyadic `[1,2,4,8,16]` rows / rank | 341 / 256 |
| rank gain, complete 8 beside 16 | 0 |
| rank gain, complete 15 beside 16 | 224 |
| coarse `[1,2,4,8]` null dimension | 960 |
| maximum pulse of exact null witness | 0.0 |
| maximum difference between two secret pulse vectors | `1.110e-16` |
| transpose/finite-difference relative error | `2.097e-10` |

Random-image, rank-192 comparison:

| design | unseen energy | condition number | noise excess |
|---|---:|---:|---:|
| pixel raster | 0.813239 | 1.000 | see JSON |
| random orthonormal | 0.810720 | 1.000 | see JSON |
| addressed box | 0.811538 | 1.64 | see JSON |

The isotropic prediction is 0.812500.

## Gate 1 - `ADDRESS_POLICY_HELPS_STRUCTURED_WORLDS_BUT_GLOBAL_ORACLE_WINS`

Relative held-out MSE after 64 scalar measurements:

| policy | MSE |
|---|---:|
| covariance-guided local lens | 0.101677 |
| random addressed local lens | 0.125617 |
| fixed fine pixels | 0.141497 |
| random global masks | 0.124620 |
| PCA global oracle | 0.043137 |

The guided lens improved by 19.06% versus random local addresses and 28.14% versus fixed fine pixels. Seeds 1 through 5 all passed the original Gate-1 thresholds; their gains versus random address ranged from 17.29% to 32.05%.

## Gate 2 - `ADDRESS_AND_PULSE_HISTORY_IDENTIFY_CONTROLLED_HIDDEN_LAW`

Default-seed result after 12 pulses over 192 trials:

| policy | exact law | direction | entropy bits |
|---|---:|---:|---:|
| fixed | 0.594 | 0.620 | 0.967 |
| random | 0.776 | 0.859 | 0.743 |
| active scale 8 | 1.000 | 1.000 | 0.000 |
| active multiscale | 1.000 | 1.000 | 0.000 |

Active multiscale address counts were concentrated at scales 2 and 16. Since active scale 8 also reached 1.000, this experiment does not establish that multiple scales are necessary.

## Gate 2 robustness - `MECHANISM_REPLICATES_LOCKED_MARGIN_DOES_NOT`

| seeds 1..10 | mean | minimum | maximum |
|---|---:|---:|---:|
| active exact accuracy | 1.000 | 1.000 | 1.000 |
| random exact accuracy | 0.796 | 0.760 | 0.828 |
| active minus random | 0.204 | 0.172 | 0.240 |

The original complete gate passed 5/10 seeds because it demanded an active-minus-random margin of at least 0.20 on each run. The threshold remains unchanged.


## Gate 3 - `READ_WRITE_SEPARATES_STATE_EXCITATION_FROM_PERSISTENT_OPERATOR_CHANGE`

Default-seed Gate 3A:

| measurement | result |
|---|---:|
| read-only max law spread | 0 |
| equal-energy uniform-write spread | 0 |
| localized-write max law spread | 0.680000 |
| active exact identification | 1.000 |
| random addressed identification | 0.355 |

Default-seed Gate 3B, after fast-state erasure and removal of the write identity:

| measurement | result |
|---|---:|
| no-operator-write max law spread | 0 |
| persistent operator-write spread | 0.075605 |
| noise-aware active accuracy | 1.000 |
| raw predictive-variance accuracy | 0.367 |
| random accuracy | 0.480 |

The two writes answer different questions. Gate 3A perturbs state so the unchanged operator becomes visible. Gate 3B changes a local operator parameter, then demonstrates that the change remains in later responses after the transient is removed.

## Gate 3 robustness - `READ_WRITE_MECHANISM_REPLICATES_10_OF_10`

| seeds 1..10 | result |
|---|---:|
| Gate 3A pass count | 10 / 10 |
| active accuracy mean / minimum | 1.000 / 1.000 |
| random accuracy mean | 0.380 |
| minimum active-minus-random margin | 0.586 |
| Gate 3B pass count | 10 / 10 |
| noise-aware accuracy mean / minimum | 1.000 / 1.000 |
| raw-variance accuracy mean | 0.411 |
| random accuracy mean | 0.432 |
| minimum noise-aware-minus-random margin | 0.516 |

These are finite controlled hidden-law families. The result is an observability mechanism, not unknown-PDE discovery or a neuronal-learning result.


## Gate 4 - `RANK_OPENS_BUT_1UV_SOMA_IDENTITY_GATE_FAILS`

Pinned human L2/3 cell 1125:

| quantity | result |
|---|---:|
| morphology nodes | 12,632 |
| eligible dendritic sections | 182 |
| hidden local leak parameters | 12 |
| candidate addressed probes | 144 |
| probe budget | 12 |
| analytic vs exact 10% sensitivity relative error | 0.0002 |

At 1 uV RMS soma noise:

| policy | numerical rank | noise-visible rank | exact hidden-section identity |
|---|---:|---:|---:|
| fixed one address, varying scale | 2 | 1 | 0.372 |
| active point-only | 12 | 7 | 0.761 |
| active multiscale | 12 | 7 | 0.779 |
| random multiscale | 9 median | 4 median | 0.539 mean (32 post-result subsets) |

Active multiscale smallest singular value was `6.3980e-05 mV`; the random
multiscale median was effectively rank-deficient. The active 12-probe sequence
used 3 point probes, 2 radius-35-um probes, and 7 radius-110-um probes.

The original locked exact-identification requirement was >=0.90 at 1 uV.
It failed and is not revised.

## Gate 4 post-result audit - `ADDRESS_OPENS_FULL_RANK_BUT_SOMA_NOISE_LIMITS_IDENTITY`

Across ten new observation-noise seeds at 1 uV RMS:

| metric | result |
|---|---:|
| active accuracy mean | 0.778 |
| active accuracy min / max | 0.772 / 0.783 |
| original >=0.90 passes | 0 / 10 |
| random-subset accuracy mean | 0.539 |

Post-result active noise sweep:

| soma noise RMS | mean exact identity |
|---:|---:|
| 0.10 uV | 1.000 |
| 0.25 uV | 0.977 |
| 0.50 uV | 0.905 |
| 1.00 uV | 0.777 |
| 2.00 uV | 0.543 |
| 5.00 uV | 0.258 |

A flattened-radius attacker also retained full rank; its smallest active singular
value was `1.3080e-04 mV`, larger than with the biological radius profile.
So this passive assay does not support radius heterogeneity as the source of
the observability gain.


## Gate 5 - `HUMAN_NMDA_RESHAPES_BUT_DOES_NOT_RESCUE_SOMA_OBSERVABILITY`

Same passive-selected 12 probe addresses in every condition:

| condition | numerical rank | visible @ 1 uV | smallest singular (mV) | exact identity |
|---|---:|---:|---:|---:|
| PASSIVE_CURRENT | 12 | 7 | 6.3980e-05 | 0.776 |
| AMPA_ONLY | 12 | 7 | 5.2641e-05 | 0.713 |
| FROZEN_NMDA | 12 | 7 | 5.2641e-05 | 0.713 |
| HUMAN_NMDA | 12 | 7 | 5.4218e-05 | 0.724 |

HUMAN versus frozen:

| quantity | result |
|---|---:|
| visible-rank gain | +0 |
| identity gain | +0.011 |
| median passive-weak gain | 1.030x |
| median passive-strong gain | 1.036x |
| weak/strong selectivity | 0.994x |

Controls:

- AMPA_ONLY and frozen NMDA discrepancy: `8.674e-19 mV`;
- HUMAN analytic implicit derivative vs direct 10% hidden-leak perturbation:
  relative error `0.0002`.

The locked rescue requirements were not met. HUMAN magnesium feedback reshapes
the sensitivity matrix slightly but does not selectively lift the five weak
Gate-4 directions or add a noise-visible dimension.


## Gate 6 - `PERSISTENT_WRITE_CHANGES_FUTURE_SENSING_BUT_MULTISCALE_NOT_EARNED`

Across 20 seeds:

| policy | paid probes | pre-prediction MSE | onset probes | repeated paid looks |
|---|---:|---:|---:|---:|
| ACTIVE_WRITE_MULTISCALE | 42.00 | 0.001563 | 4.375 | 0.00 |
| ACTIVE_NOWRITE_MULTISCALE | 280.00 | 0.006667 | 4.375 | 56.00 |
| ACTIVE_WRITE_FINE | 42.00 | 0.001563 | 4.375 | 0.00 |
| RANDOM_WRITE_MULTISCALE | 56.60 | 0.001729 | 5.08 | 1.60 |

Active-write localization accuracy was 1.000.

Write versus no-write:

| comparison | ratio |
|---|---:|
| pre-prediction MSE | 0.234 |
| paid probes | 0.150 |
| repeated paid looks | 0.000 |

The original multiscale requirement failed because raw predictive variance used
the same onset-search cost as fine-only: ratio 1.000.

## Gate 6B post-result audit - `INFORMATION_GAIN_UNLOCKS_COARSE_TO_FINE_SEARCH_POSTHOC`

Expected Bayesian information gain replaced raw predictive variance only after
the Gate-6 scale failure.

| selector | onset probes |
|---|---:|
| raw predictive variance, multiscale | 4.375 |
| fine-only | 4.375 |
| expected information gain, multiscale | 3.000 |

Information/fine cost ratio: **0.686**. Localization remained **1.000**.

The information selector used scale sequence **16 -> 8 -> 4 on all 160/160
audited onset searches**.

This does not retroactively pass the Gate-6 preregistration; it is a new
post-result mechanism.


## Gate 6C post-result audit - `INTERMEDIATE_LOCAL_WRITE_TIMESCALE_MINIMIZES_PROBES_BUT_HISTORY_ATTACKER_REMOVES_FAST_WRITE_PENALTY`

WRITE now occurs between pulses in the same addressed field:

```text
x <- x + alpha * residual * h / ||h||^2
alpha = 1 - exp(-1/tau_write)
```

Across 80 trials per timescale:

| tau / probe | local same-field probes | history-replay probes |
|---:|---:|---:|
| 0 | 9.750 | 5.625 |
| 0.25 | 9.750 | 5.625 |
| 0.5 | 9.638 | 5.700 |
| **1** | **8.425** | 6.075 |
| 2 | 10.175 | 7.613 |
| 4 | 14.100 | 10.200 |
| 8 | 18.900 | 14.888 |
| 16 | 22.913 | 21.225 |
| inf | 24.000 | 24.000 |

For the local writer, instant/best cost is 1.157 and no-write/best is 2.849.
Localization remains 1.000 at every tested timescale.

Fast local WRITE therefore does not destroy identity evidence in this high-SNR
toy. Its penalty is HOME overshoot from overlapping coarse/fine
backprojections. The pulse-history estimator attacker removes that fast-write
penalty and is best at instantaneous update.
