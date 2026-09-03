# Morgue and quarantine

## Killed here

### “The multiscale lens compresses a random image”

No. At matched rank, an IID Gaussian image gives the same expected captured energy to every row-space orientation. Gate 0 obtains a 0.002519 spread around the exact isotropic prediction. An arbitrary 1024-degree image still needs rank 1024 for unique noiseless reconstruction.

### “All scales accumulate independent evidence”

No. Complete dyadic partitions are nested. Scales 1, 2, 4, 8 and 16 supply 341 rows but rank only 256. Coarse rows may still improve noisy estimates, but they do not create new noiseless directions.

### “Coprime lenses are automatically the best sensing policy”

No. Coprimality minimizes the exact shared partition subspace for a complete pair. Sequential performance also depends on measurement cost, noise, conditioning, the image prior, and address order. Gate 1 selects a mixture of commensurate and incommensurate scales rather than obeying a coprime-only rule.

### “Multiscale sensing was necessary for Gate 2”

No. Active scale 8 and active multiscale both achieved 100% on the controlled 64-law family. Gate 2 earns active address selection. It does not yet earn a multiscale necessity claim.

### “Gate 2’s 0.20 advantage is fully robust”

No. The exact active result replicated 10/10, but the originally locked active-minus-random margin was below 0.20 on five seeds. The threshold is not revised after seeing this.



### “Any write reveals hidden dynamics”

No. Gate 3A's equal-energy uniform write is an exact null for all four transport laws. The useful write is localized because it excites spatial structure on which the laws act differently.

### “Active means probe the largest predicted signal difference”

No. Gate 3B deliberately makes fine probes noisy. Raw predictive-variance selection averages 0.411 exact accuracy across the robustness seeds, slightly below random at 0.432. A noise-aware separation criterion reaches 1.000.

### “A diagnostic write and a structural write are the same thing”

No. Gate 3A changes only fast state and leaves the operator fixed. Gate 3B changes a local operator parameter; its signature survives erasure of the fast field and write log. Keeping these separate is now a standing obligation.
\n

### “Full rank means the soma can read the hidden dendritic state”

No. Gate 4 active stimulation opens the real-morphology passive Jacobian to
12/12 numerical rank, yet at the locked 1-uV RMS soma-noise ruler only 7/12
singular directions are visible and exact hidden-section identity is about
0.78. Algebraic observability and reliable readout are different gates.

### “Multiscale stimulation is necessary on the real tree”

No. Active point-only probes already reach full rank and 0.761 identity versus
0.779 for multiscale. Gate 4 earns address selection; cluster scale adds only a
modest gain in this assay.

### “The biological radius profile creates the observability gain”

Not supported. Flattening every dendritic radius to the real-cell median keeps
full rank and increases the smallest singular value in this passive model.
Topology/address placement survive as sufficient ingredients for the measured
rank opening.

## Earlier kills carried forward

- The original ECG graph remains V21's quantized aliasing map plus variance thermostat, not a lag/skew operator.
- Phase-as-memory without unknown-address retrieval remains killed by V21's no-probe/wrong-probe null.
- Passive geometry does not manufacture one-way propagation; V20/V13's reciprocity wall remains.
- Geometry or a large state alone is not evidence of biological computation.

## Quarantined bets

- A persistent read/write field can develop a useful morphology.
- A learned or evolved local law can turn pulse history into useful structural change.
- The address bus can become part of the same geometry it reads and writes.
- A physical reciprocal substrate can perform the relevant adjoint experiment.
- Any of this illuminates a biological neuron beyond analogy.

Each bet requires a new experiment. None is promoted by the current three gates.


### “HUMAN NMDA turns the soma into a detailed branch-state observer”

Killed by Gate 5 in the current reduced bridge.

Using the same 12 passive-selected addresses, HUMAN NMDA leaves the
noise-visible rank at 7/12 and improves exact identity only from 0.713 to 0.724
relative to the rest-matched frozen-block control. The five weak passive
directions gain 1.030x while strong directions gain 1.036x, so there is no
selective rescue.

This does not kill dendritic NMDA computation. It kills the narrower attempted
use of that nonlinearity as a central soma-observability rescue in this assay.



### “Raw predictive variance is enough to discover coarse-to-fine sensing”

Killed by Gate 6.

With equal detector noise, raw predictive variance chose high-amplitude fine
4x4 questions and required 4.375 probes per anomaly onset — exactly the
fine-only cost. Merely offering multiple scales did nothing.

The post-result expected-information selector did use the hierarchy, but that
is a different policy and is recorded separately as Gate 6B.

### “A successful read is memory even if it does not change future state”

Killed by the Gate-6 no-write attacker.

The no-write observer localized the same anomalies with the same active search,
but because the result did not persist in the predicted spatial field it paid
280 probes instead of 42 and reopened paid search on 56 continuing-anomaly
steps instead of zero.

For this architecture, memory earns its name only when the write changes the
next prediction and therefore the next sensing action.



### “Fast same-field WRITE necessarily destroys the next measurement”

Not in Gate 6C's current high-SNR assay.

Moving WRITE between the 16 -> 8 -> 4 probes and sweeping its time constant
leaves anomaly localization at 1.000 for every tested value from instantaneous
WRITE through no WRITE.

Fast local WRITE is still worse than an intermediate rate, but for a different
reason: nested local backprojections over-correct the spatial field and push the
free HOME residual negative while the anomaly is still present. That creates
extra verification pulses.

So the literal “broken copy destroys identity evidence” claim is not earned
here.

### “The intermediate write timescale is a universal requirement”

No.

A HISTORY_REPLAY attacker stores every pulse equation and recomputes a
minimum-norm field consistent with the accumulated history. It removes the
fast-write penalty entirely and is best at instantaneous update (5.625 probes
versus 8.425 at the best local-field timescale).

The timescale optimum is therefore conditional on refusing that extra
bookkeeping and using only local writes into the addressed spatial field.
