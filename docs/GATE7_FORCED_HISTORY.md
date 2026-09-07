# Gate 7 — finite addressed history closes the fast-write gap

Gate 6C left an architectural fork:

```text
LOCAL_SAME_FIELD
    no pulse-history bookkeeping
    instantaneous write: 9.750 paid probes

HISTORY_REPLAY
    stores every addressed measurement equation
    instantaneous write: 5.625 paid probes
```

The full-history attacker removed the fast-write penalty, but it did so by keeping
an unbounded digital ledger.  Gate 7 asks whether a *small* recent history is
enough.

## The intermediate object

At each paid read, keep only the most recent `K` pairs

```text
(address h_i, scalar pulse y_i)
```

and reconstruct the minimum-norm field consistent with that finite window.
WRITE is instantaneous (`tau_write = 0`).

This is motivated by the Takens/efference-copy side quest, but the distinction is
important: it is finite **addressed measurement history**, not a claim that the
classical autonomous Takens theorem applies to this field.

## Result

Across 10 noise seeds x 8 anomaly identities = 80 trials per condition:

| history window K | correct address: paid probes | scrambled address: paid probes |
|---:|---:|---:|
| 1 | 13.500 | 13.500 |
| 2 | 8.250 | 15.250 |
| **3** | **5.625** | **15.250** |
| 4 | 5.625 | 15.250 |
| 6 | 5.663 | 15.250 |
| 8 | 5.625 | 15.250 |
| 12 | 5.625 | 15.250 |
| 16 | 5.625 | 15.250 |
| 32 | 5.663 | 15.250 |
| 64 | 5.625 | 15.250 |

Baselines:

```text
LOCAL_SAME_FIELD, no history      9.750 probes
FULL HISTORY_REPLAY               5.625 probes
FINITE HISTORY, K=3               5.625 probes
```

All correct-address conditions retain 1.000 localization accuracy.

At `K=3`, negative HOME overshoot falls from **3.4875 triggers/event** in the
local same-field writer to **0**, matching full replay.  Repeated paid steps fall
from **4.575** to **0.875**, again matching full replay.

## Why three is not mystical

The active search in this locked task is itself a three-pulse hierarchy:

```text
16 -> 8 -> 4
```

So the clean interpretation is not "embedding dimension three discovered the
brain."  It is that the system only needs to retain the **currently independent
measurement constraints that define the present estimate**.  Older constraints
are redundant for this stationary eight-way onset task.

That makes Gate 7 closer to a finite sufficient-statistic result than a direct
Takens theorem result.

## The address attacker is the important part

The paired attacker preserves the scalar values and the history length but
attaches every pulse to the wrong same-scale lens address.

At `K >= 2` it stays at **15.250 probes**, with ~**1.75 negative HOME triggers**
and final MSE near **0.0201**, about five times the correctly addressed estimate.
A 64-pulse wrong-address history is worse than a three-pulse correct-address
history.

So the useful object is not

```text
history of scalar pulses
```

but

```text
history of (what I did / where I looked, what came back)
```

That is exactly where the efference-copy analogy becomes more precise.  The
system must retain enough information about its own intervention to interpret
the return.

## Relation to Takens

A classical delay vector is

```text
[y_t, y_(t-tau), y_(t-2tau), ...]
```

Gate 7 instead uses

```text
[(h_t,y_t), (h_(t-1),y_(t-1)), ...]
```

where `h_t` is chosen actively and may change each step.  The closest mathematical
family is therefore **forced / input-output state reconstruction**, not the
simplest autonomous delay embedding.

The JelloBrain S21 experiment complements this result: there, action identity
was indispensable for predicting a self-generated material change, while an
8-step temporal history provided a smaller additional improvement.  Here,
intervention **address** is indispensable, and only three current constraints are
needed to match perfect bookkeeping.

Together the experiments suggest a compact object:

> **a finite history of action/address + consequence can function as the reference
> needed to distinguish self-caused change from innovation.**

That is the computational bridge worth keeping.  It is stronger and safer than
"Takens = efference copy."

## AIS boundary

Nothing here rescues the old ~190 nm AIS-grating hypothesis.  Gate 7 needs
millisecond/task-scale history tied to intervention identity.  The periodic
actin/spectrin scaffold remains on the wrong spatial/time scale for this role and
has no established function as a delay grating.

AIS can still be used as an analogy for the **launch boundary**.  The temporal /
predictive reference should remain a separate mechanism unless biological data
says otherwise.

## Next gate

The current result used explicit digital `(address,pulse)` bookkeeping.  The next
harder question is whether the *substrate itself* can carry the same finite
reference without an external list.

A clean Gate 8 would replace the three-item ledger with a small recurrent or
spatial state driven only by

```text
address issued -> pulse returned
```

then test whether that state still removes the fast-write penalty and whether
scrambling the action/address input destroys the benefit.

That would finally move from **finite bookkeeping** toward **embodied temporal
state** — the place where a dendritic/cable or cerebellar-filter analogy becomes
experimentally meaningful.

Frozen receipt: [`results/gate7/gate7_summary.json`](../results/gate7/gate7_summary.json).
