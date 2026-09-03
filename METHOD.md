# Method and locked obligations

## Object

The hidden field is a square piecewise-constant image `x in R^(n*n)`. A lens with side `r` partitions the unit square into `r*r` cells. Address `(r, i, j)` returns the exact area average of `x` over cell `(i,j)`:

```text
y_(r,i,j) = h_(r,i,j)^T x.
```

For non-divisor resolutions, source-pixel and lens-cell boundaries are integrated by exact interval overlap. Every mask is nonnegative and sums to one.

The current Perception Lab `ImageToVectorNode` uses OpenCV `INTER_AREA`, `ceil(sqrt(output_dim))`, truncation for non-square dimensions, and per-frame maximum normalization. Those implementation choices are intentionally not imported. V24 needs an explicit linear measurement operator whose rank, nullspace, and transpose can be audited.

## Gate 0 - measurement algebra

Locked requirements:

1. selected two-scale ranks must match `r^2 + s^2 - gcd(r,s)^2`;
2. complete 8 and 16 lenses must add no rank beyond scale 16;
3. complete 15 and 16 lenses must add rank beyond scale 16;
4. an explicit checkerboard null witness must produce pulses below `1e-12` on scales 1, 2, 4, 8;
5. two different secrets separated by that witness must remain pulse-identical below `1e-12`;
6. the transpose gradient must match a central finite difference below `1e-7` relative error;
7. raster, random-orthonormal, and addressed-box designs must each have rank 192;
8. on IID Gaussian images, their mean unobserved-energy spread must be below `0.015`.

The IID image is a mandatory null. For an isotropic distribution, any rank-`m` orthogonal projection captures the same expected fraction `m/N` of energy. A win by a measurement orientation would indicate leakage, unequal rank, or an unfair normalization.

## Gate 1 - structured ensemble

Training and held-out worlds are independently generated from the same controlled family: one to four anisotropic Gaussian blobs, an optional hard-edged rectangle, and small pixel noise.

A 128-component PCA covariance model is fit on 768 training worlds. The covariance-guided lens greedily chooses from all local box averages at scales 1 through 16 to maximize expected reduction of posterior covariance. It chooses the entire sequence before seeing a held-out secret.

Comparators:

- random addressed box averages, sampling scales uniformly;
- uniformly selected fine pixels;
- random orthonormal global masks;
- top PCA directions as an intentionally stronger global oracle.

All methods receive the same scalar count and observation-noise standard deviation. Reconstruction uses the same learned Gaussian prior.

Locked Gate-1 requirements at 64 measurements:

1. guided local lens improves relative MSE by at least 10% over random addresses;
2. guided local lens improves by at least 10% over fixed fine pixels;
3. the PCA global oracle remains no worse than the local lens;
4. the guided design actually selects more than one scale.

## Gate 2 - controlled hidden-law identification

The initial Gaussian field and the model family are public. Only the identity of the law is hidden:

```text
8 directions x 2 speeds x 2 diffusion rates x 2 decay rates = 64 laws.
```

At each of 12 time steps, each observer receives one scalar with Gaussian noise. Bayesian belief is exact over the finite table. The active policy selects the address with maximum posterior predictive variance - the place where surviving hypotheses disagree most.

Comparators:

- one fixed scale-8 address;
- random scale and address;
- active address at scale 8 only;
- active scale and address over the full candidate family.

Original locked requirements:

1. multiscale active exact accuracy at least 0.90;
2. active exceeds random by at least 0.20;
3. active exceeds fixed by at least 0.40;
4. final active entropy at most 0.5 bits;
5. multiscale active is not more than 0.02 worse than active scale 8.

The default seed passed. A post-result seeds 1 through 10 audit retained the same thresholds. Active exact identification remained 1.000 on all ten seeds, but the 0.20 active-minus-random condition held on only five. This is reported as `MECHANISM_REPLICATES_LOCKED_MARGIN_DOES_NOT`.

## Gate 3 - READ + WRITE

Gate 3 separates **state excitation** from **operator modification**.

### Gate 3A - state write reveals hidden transport

The hidden operator is one of four periodic shift-plus-diffusion laws. From the zero state all four are exactly indistinguishable. Two writes have identical L2 energy:

- a uniform field, which every transport law preserves;
- one localized impulse.

Reads use the same addressed box-average family at scales 4, 8, and 16. A Bayesian active policy chooses the address with maximum posterior predictive separation. Random addressed reads are the attacker.

Locked requirements:

1. read-only law spread below `1e-13`;
2. uniform-write law spread below `1e-13`;
3. localized-write law spread above `0.25`;
4. active exact identification at least `0.99`;
5. active exceeds random by at least `0.35`.

The write changes `x`, not the transport law. This earns diagnostic excitation only.

### Gate 3B - persistent operator write survives reset

One of four 4 x 4 regions receives a persistent change in the local retention coefficient. The original fast state is erased and the observer is not told the write address. Every hypothesis is then loaded with the same public uniform diagnostic field.

Reads use scales 2, 4, 8, and 16. Detector noise is deliberately heteroscedastic and known: fine lenses are much noisier. Three policies are compared:

- random address;
- raw posterior predictive variance;
- noise-aware predictive separation, variance divided by detector-noise variance.

Locked requirements:

1. no-operator-write law spread below `1e-13`;
2. persistent-write law spread above `0.005`;
3. noise-aware exact identification at least `0.99`;
4. noise-aware exceeds random by at least `0.45`;
5. raw predictive variance does not exceed `0.75` exact accuracy.

The raw-variance attacker matters: an active policy that ignores measurement quality can be worse than random.

### Gate 3 robustness

Seeds 1 through 10 rerun both halves with all original thresholds unchanged. Passing requires Gate 3A and Gate 3B to pass on every seed.

## Gate 4 - real cell-1125 soma observability

The source morphology is pinned to `ido4848/FCI` commit
`75ad8b4d81a7f51bf888b30650c543592340db06`, file
`2013_03_06_cell11_1125_H41_06.asc`.

MorphIO supplies the point tree. Gate 4 builds a passive DC morphology-graph
cable directly from edge length, radius and topology with `Ra=150 ohm cm` and
`Rm=20,000 ohm cm^2`. This is not the released FCI NEURON model.

Twelve deterministic dendritic sections are hidden-parameter locations. Hidden
parameter `p_j=1` means a 10% local leak-density increase on section `j`.

A probe injects 0.10 nA total current at one of 48 dendritic centers, using
geodesic cluster radius 0, 35 or 110 um. Thus there are 144 candidate probes.
Only soma voltage is read.

For passive matrix `A`, probe `b_i`, soma selector `e_s`, and leak
perturbation `D_j`:

```text
x_i = A^-1 b_i
J_ij = -e_s^T A^-1 D_j A^-1 b_i
```

The 12-probe active policy greedily maximizes regularized Fisher log-determinant.
Comparators are one fixed address with varying scales, active point-only probes,
and random point/multiscale subsets.

Original locked requirements included:

1. >1000 real morphology nodes;
2. active multiscale reaches numerical rank 12/12;
3. active smallest singular value exceeds random-multiscale median by >=20%;
4. exact hidden-section fingerprint accuracy >=0.90 at 0.001 mV RMS soma noise;
5. fixed-address rank <=3;
6. the analytic 10% sensitivity matches a direct finite difference within 8%.

Requirement 4 failed. It remains locked.

The post-result audit then varied only observation-noise seeds and noise
amplitude. It is explicitly not a replacement gate.

## Measurement normalization

Physical lens outputs are area averages. Rank is invariant to nonzero row scaling. For Gate 0's noise/conditioning comparison only, every design row is normalized to unit Euclidean energy; this gives box masks a generous equal-signal-variance comparison against global masks. The JSON receipt marks this explicitly.

## Claim discipline

- Rank is not reconstruction quality.
- Redundant coarse measurements may improve noise tolerance even when they add no noiseless rank.
- “Coprime” controls exact overlap for a two-partition model; it is not automatically an optimal sequential policy.
- A software transpose is not a same-device physical adjoint.
- Identifying one member of a known finite table is not discovering an arbitrary dynamical law.
- An image-space laboratory is not evidence about biological neurons.


## Gate 6 - PerceptionLab persistent spatial memory

World: a 32x32 structured field with known toroidal +1-pixel x transport per
step. Eight supplied 4x4 anomaly prototypes can appear. The schedule uses eight
cycles of 4 OFF steps followed by 8 ON steps.

The observer gets a free whole-field HOME average with Gaussian noise sigma
0.001. Paid square lenses use scales 16, 8 and 4 with noise sigma 0.002.

A HOME residual above 0.004 opens a local search. The original Gate-6 active
selector maximizes predictive scalar variance. Once a prototype posterior
exceeds 0.995, write-capable policies insert the recognized prototype into the
predicted spatial field. That field is transported on the next step.

The primary attacker is identical except the write is erased.

Locked comparisons include prediction MSE, paid-probe count, repeated paid
looks during a continuing anomaly, localization accuracy, and a multiscale
onset-cost requirement versus fine-only search.

The multiscale requirement failed and remains failed.

Gate 6B was introduced only afterward. It numerically integrates expected
Bayesian information gain for each candidate noisy lens and chooses the lens
with maximal mutual information about prototype identity. Gate 6B is a
post-result policy audit, not a replacement preregistration.


## Gate 6C - post-result same-field write-timescale audit

Gate 6C is not a new preregistered gate. It isolates the temporal objection to
Gate 6's persistent write.

The anomaly remains present for eight steps. Paid lenses use the same 16, 8 and
4 geometry and noise levels as Gate 6. Probe choice uses the high-SNR discrete
partition form of Gate 6B information gain.

After every paid residual pulse `y = h^T(world - x)`, the local writer updates
the same addressed field before the next probe:

```text
x <- x + alpha y h / ||h||^2
alpha = 1 - exp(-1/tau_write)
```

The sweep is `tau/probe = 0, .25, .5, 1, 2, 4, 8, 16, inf`, over ten noise
seeds and all eight anomaly identities.

The attacker `HISTORY_REPLAY` stores absolute pulse equations and repeatedly
solves the minimum-norm field consistent with that history before applying the
same alpha. It is intentionally a conventional state-estimation/bookkeeping
control rather than a same-field mechanism.

Reported metrics are paid probes per eight-step event, repeated paid steps,
negative HOME triggers, final field error, and localization accuracy.
