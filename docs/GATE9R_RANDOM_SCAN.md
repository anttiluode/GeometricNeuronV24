# Gate 9R — after removing the policy shortcut, history matters

Gate 9 initially looked stronger than it was: even `TAP1` matched full replay.
That was a shortcut.  The adaptive `16 -> 8 -> 4` search chooses its next address
from previous evidence, so the *current address itself* leaks much of the search
history to a memoryless decoder.

Gate 9R removes that leak.

## Random-order scan

Every trial probes all eight fine `4x4` lenses in a fresh random permutation.
The order is independent of hidden anomaly identity.  Therefore the current
address no longer summarizes previous evidence.

The returned pulse is still measured relative to the current internal estimate,
so the observer must retain enough information about earlier self-generated
measurements/writes to reconstruct the target.

## Result

80 held-out trials:

| state | hidden identity accuracy | field MSE |
|---|---:|---:|
| full addressed HISTORY_REPLAY | **1.000** | 4.92e-7 |
| current addressed pulse only (`TAP1`) | 0.2875 | 0.00768 |
| two addressed taps | 0.4500 | 0.00607 |
| three addressed taps | 0.5625 | 0.00483 |
| four-timescale addressed leaky bank | **1.000** | 9.29e-7 |
| four-timescale scalar-only bank | **0.125** | 0.00877 |

Chance identity accuracy is `1/8 = 0.125`.

So the original Gate 9 shortcut is killed.  Once current-address policy leakage
is removed, a single present measurement is not enough, explicit temporal depth
helps progressively, and a small multi-timescale **addressed** trace bank can
match full bookkeeping on this stationary task.

The scalar-only temporal bank is exactly at chance.  History without knowing
which intervention produced each return is not enough.

## What this does and does not say about Takens

This is now much closer to the intuition that motivated the side quest:

```text
what I did / where I pinged
        +
what came back
        +
short history of those pairs
        ->
reconstruct enough hidden state
```

But this task is still stationary.  The hidden anomaly does not evolve while it
is being reconstructed.  Therefore calling the result a Takens phenomenon would
still be too strong.  In a static inverse problem, temporal traces can simply be
an efficient memory of accumulated constraints.

A genuine Takens/efference test needs a **moving hidden state** whose present
state cannot be recovered from one present observation, plus known self-actions
that alter either the state or the observation operator.  Then the question is
whether a delay history conditioned on those actions reconstructs enough state
to predict the next return and expose an unreported perturbation as innovation.

That is Gate 10.

## Biological boundary

The computational motif is compatible with many substrates: recurrent state,
cerebellar-like adaptive filters, dendritic temporal filtering, synaptic traces,
or explicit digital memory.  Gate 9R does not identify which biology implements
it.

It also does not rehabilitate the ~190 nm AIS periodic scaffold as a delay line.
The AIS remains relevant as a launch/excitability boundary; the temporal
reference is a separate computational object until evidence connects them.

Frozen receipt: [`results/gate9r/gate9r_summary.json`](../results/gate9r/gate9r_summary.json).
