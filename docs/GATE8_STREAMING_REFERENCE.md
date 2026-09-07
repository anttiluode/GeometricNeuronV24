# Gate 8 — the reference can be a subspace, not a copy

Gate 7 removed the need for unbounded HISTORY_REPLAY: the current task needed
only the most recent three correctly addressed pulse equations.  But it still
kept those three equations as a chronological list.

Gate 8 removes the list.

## Streaming state

For each addressed measurement

```text
y = h^T x
```

retain only:

```text
x_hat   current estimate
Q       orthonormal basis of independent measurement directions
```

For a new probe address `h`, compute the part that is genuinely new relative to
the retained measurement subspace:

```text
q = h - Q Q^T h
```

and update only along that new direction:

```text
x_hat <- x_hat + (y - h^T x_hat) q / ||q||^2
```

Then `q/||q||` becomes another basis direction.  Old pulse values and old
addresses are not retained as a list.  Their effect survives only in the current
estimate and the geometry of `Q`.

This is plain streaming linear algebra, not a proposed biological algorithm.

## Result

80 trials per condition:

| retained independent directions | paid probes | repeated paid steps | negative HOME triggers |
|---:|---:|---:|---:|
| 1 | 13.500 | 3.500 | 0.000 |
| 2 | 8.250 | 1.750 | 0.000 |
| **3** | **5.625** | **0.875** | **0.000** |
| 4 | 5.625 | 0.875 | 0.000 |
| 8 | 5.638 | 0.888 | 0.013 |

Baselines:

```text
local same-field, no reference state    9.750 probes
full unbounded HISTORY_REPLAY            5.625 probes
rank-3 streaming reference               5.625 probes
```

The streaming state itself settles at rank **3** even when allowed rank 4 or 8.
That is exactly the number of independent addressed measurements in the locked
`16 -> 8 -> 4` search.

So Gate 7's result was not really about preserving "the last three moments."
The stronger statement is:

> **Preserve the three independent intervention directions that currently define
> the estimate.**

Time order is secondary here.  Geometry of intervention is primary.

## Wrong-address attacker

The scalar pulses are left unchanged, but each is paired with the wrong
same-scale probe mask.

At rank 3 and above:

```text
correct address      5.625 probes, final MSE ~0.00405
wrong address       15.250 probes, final MSE ~0.0201
```

More state does not rescue a wrong efference/address copy.

This sharpens the connection to corollary discharge.  The useful reference is
not simply "a memory of what came back."  It must remain paired with enough
information about **what the system itself did** to make the return interpretable.

## What happened to the Takens intuition?

It did not disappear; it became more precise.

Takens-style delay coordinates say that a suitable history of partial
observations can supply coordinates for hidden state.  In this active observer,
however, the observation operator itself changes because the system chooses
`h_t`.

Gate 8 says the compact sufficient state is governed by the rank of the
**intervention-conditioned observability directions**, not merely the number of
time delays.

That is why a more useful mathematical phrase for this line is now:

> **controlled / forced observability with a finite internal reference state.**

The JelloBrain S21 result complements this: in its plastic world, the known
write/action copy produced the large prediction gain; temporal delay history
added a smaller correction and had a finite optimum near eight samples.

So there are two regimes already visible:

```text
GeometricNeuron Gate 8
    changing probe address dominates
    -> remember independent observation directions

JelloBrain S21
    hidden plastic state evolves under actions
    -> current action dominates, short history adds state information
```

Both are versions of the same larger problem: reconstruct enough hidden state
from partial observations **conditioned on one's own interventions**.

## The old copy intuition, revised

The system does not need:

```text
complete duplicate of BEFORE
```

and it does not necessarily need:

```text
complete chronological history
```

In this linear toy it needs only:

```text
current estimate
+
protected intervention subspace
```

That is an economical counterfactual reference.  It remembers what must remain
true when the next write occurs.

This is close to the recurring JelloBrain question:

> what may I change without damaging something else?

The basis `Q` literally marks directions whose already-measured consequences
must be preserved; a new correction is projected into the orthogonal innovation
direction.

## Biological boundary

Do not read `Q` as a claim that dendrites perform Gram-Schmidt.  Biology could
approximate protected directions through compartmentation, mixed temporal
filters, recurrent circuitry, inhibitory gating, synaptic eligibility, or other
mechanisms.  Gate 8 only identifies the computational object that such a
mechanism would need to approximate.

Likewise the ~190 nm AIS actin/spectrin periodicity remains outside the delay
story.  AIS is still relevant as a launch and plastic excitability boundary, not
as an established Takens grating.

Frozen receipt: [`results/gate8/gate8_summary.json`](../results/gate8/gate8_summary.json).

## Next

The interesting next step is no longer another exact estimator.  Gate 9 should
ask whether a **physically plausible bank of leaky / delayed traces** can
approximate this rank-3 protected subspace without explicit orthogonalization.

That is the place where the old dendritic-delay intuition and cerebellar
adaptive-filter analogy become testable rather than decorative.
