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

## Measurement normalization

Physical lens outputs are area averages. Rank is invariant to nonzero row scaling. For Gate 0's noise/conditioning comparison only, every design row is normalized to unit Euclidean energy; this gives box masks a generous equal-signal-variance comparison against global masks. The JSON receipt marks this explicitly.

## Claim discipline

- Rank is not reconstruction quality.
- Redundant coarse measurements may improve noise tolerance even when they add no noiseless rank.
- “Coprime” controls exact overlap for a two-partition model; it is not automatically an optimal sequential policy.
- A software transpose is not a same-device physical adjoint.
- Identifying one member of a known finite table is not discovering an arbitrary dynamical law.
- An image-space laboratory is not evidence about biological neurons.
