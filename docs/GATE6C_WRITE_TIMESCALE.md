# Gate 6C — the write has a clock

Gate 6 established that a persistent spatial write can reduce later paid sensing,
and Gate 6B showed that information gain can produce the 16 -> 8 -> 4 lens
sequence. Both of those assays make the durable write only after an onset search
has finished.

The obvious objection is temporal:

> if READ and WRITE touch the same addressed spatial state, can a fast write
> damage the evidence that the next pulse is trying to measure?

Gate 6C moves WRITE between pulses and sweeps its time constant.

## Same-field update

Each paid lens returns a residual pulse

```text
y = h^T(world - x)
```

and immediately writes that pulse back into the same spatial field:

```text
x <- x + alpha y h / ||h||^2

alpha = 1 - exp(-1 / tau_write)
```

The probe interval is one unit. Thus `tau=0` is effectively instantaneous and
`tau=inf` is no write.

The world is one of Gate 6's eight 4x4 anomaly locations and remains present for
eight steps. HOME noise, paid-lens noise, surprise threshold, and lens geometry
match Gate 6. Probe selection uses the high-SNR partition form of Gate 6B
information gain, which reproduces 16 -> 8 -> 4 throughout this audit.

## The attacker matters

Two write rules are compared.

**LOCAL_SAME_FIELD** keeps only the spatial field. Each pulse backprojects
locally into that field. There is no separate pulse-history solver.

**HISTORY_REPLAY** is the boring estimator attacker. It stores every absolute
pulse equation and recomputes the minimum-norm field consistent with all of
them, then applies the same write-rate alpha. This is deliberately the
bookkeeping escape hatch.

## Result

Across 10 noise seeds x 8 anomaly identities = 80 trials per timescale:

| tau_write / probe interval | local same-field probes | history-replay probes |
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

For the local same-field writer:

```text
best tau / probe interval       1
best paid probes                8.425
instantaneous paid probes       9.750
no-write paid probes           24.000

instant / best                  1.157
no-write / best                 2.849
```

So there really is an intermediate-timescale minimum in this write rule.

But the failure mode is more specific than the original "broken copy" wording.

At very fast local WRITE, localization remains **1.000** at every tested
timescale. The in-flight identity evidence is not destroyed in this high-SNR
toy. What goes wrong is control: overlapping 16 -> 8 -> 4 backprojections
over-correct different nested averages, pushing the free HOME residual negative
while the anomaly is still present. Instant local WRITE produces 3.49 negative
HOME triggers per event on average.

Slow WRITE fails for the opposite reason. It changes the field too little, so
the continuing anomaly remains surprising and the full paid search keeps
reopening. With no write, all eight event steps are sensed and the cost returns
to 24 probes.

The estimator attacker changes the interpretation again. HISTORY_REPLAY removes
the fast-write penalty completely: its best point is instantaneous WRITE at
5.625 probes, with zero negative HOME triggers.

Classification:

`INTERMEDIATE_LOCAL_WRITE_TIMESCALE_MINIMIZES_PROBES_BUT_HISTORY_ATTACKER_REMOVES_FAST_WRITE_PENALTY`

Copy-worry boundary:

`IDENTITY_EVIDENCE_NOT_CORRUPTED_IN_THIS_HIGH_SNR_TOY`

## What this earns

The useful statement is narrower than "biology solved copying with two
timescales."

> **If pulse history is not stored in a separate consistency solver, a local
> spatial write can acquire a real control-timescale optimum.**

The fast side is overshoot/interference between overlapping writes. The slow
side is failure to amortize sensing.

A digital state estimator can remove the fast penalty by keeping extra
measurement history. That attacker is exactly why Gate 6C does **not** promote
the timescale optimum into a universal law.

What survives is a cleaner architectural fork:

```text
same spatial field only
        -> write rate becomes part of control stability

extra pulse-history bookkeeping
        -> fast consistent updates are possible
```

This is a post-result mechanistic audit. It does not alter Gate 6, Gate 6B, or
the biological boundary.
