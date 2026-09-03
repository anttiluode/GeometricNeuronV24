# Gate 6 — back to PerceptionLab: history changes future sensing

The neuron bridge stopped at Gate 5. Gate 6 returns Geometric Neuron to the
stronger synthetic question that existed before the biological analogy:

> Can an addressable spatial system remember what it learned so that the next
> sensory action is different?

The workflow is now literal:

```text
remember
   |
predict
   |
HOME pulse
   |
surprise?
   |
pay for an addressed look
   |
identify local cause
   |
WRITE correction into spatial memory
   |
transport that memory forward
   |
future HOME pulse changes
```

## World

A structured 32 x 32 field translates one pixel per time step.

Eight possible 4 x 4 anomaly prototypes can appear. Each anomaly persists for
eight steps, followed by an off interval. The observer knows the prototype
dictionary and the global translation law; it does **not** know which anomaly
appeared.

The whole experiment is therefore about sensing policy and durable spatial
memory, not dictionary learning.

## Sensors

The system receives one free scalar **HOME** pulse: whole-field average.

HOME can say that total mass changed but cannot localize which of the eight
possible patches caused it.

Paid lenses are square area averages at three scales:

```text
16 x 16
 8 x 8
 4 x 4
```

A local search can move both address and scale.

## Durable state

The only durable perceptual state is the predicted spatial field.

When a hidden prototype is localized, the recognized prototype is written into
that field. On the next step the field is transported forward with the known
global motion.

That write is the key intervention. The controller does not merely remember a
label in an unrelated table.

## Locked Gate-6 result

Across 20 seeds:

```text
policy                         paid probes   pre-prediction MSE   onset probes   repeated paid looks

ACTIVE_WRITE_MULTISCALE          42.00          0.001563             4.38             0.00
ACTIVE_NOWRITE_MULTISCALE       280.00          0.006667             4.38            56.00
ACTIVE_WRITE_FINE                42.00          0.001563             4.38             0.00
RANDOM_WRITE_MULTISCALE          56.60          0.001729             5.08             1.60
```

The active writer localized every onset correctly.

Relative to the identical active reader with write memory erased:

```text
prediction-MSE ratio          0.234
paid-probe ratio              0.150
repeated-look ratio           0.000
```

So the strong result is:

> **A persistent spatial write changes future sensing.**

Once the anomaly is written, subsequent predicted HOME pulses agree with the
world and the system stops paying to rediscover the same surprise.

The no-write attacker has exactly the same detector and active search policy,
but because its correction disappears it reopens the same search on every
continuing anomaly step.

Classification:

`PERSISTENT_WRITE_CHANGES_FUTURE_SENSING_BUT_MULTISCALE_NOT_EARNED`

## The first multiscale claim failed

Gate 6 had also preregistered that its raw predictive-variance selector should
use the coarse lenses efficiently.

It did not.

```text
raw variance multiscale onset search    4.375 probes
fine-only onset search                  4.375 probes
ratio                                   1.000
```

The original multiscale criterion therefore remains failed.

This was not a code failure. It exposed a policy mistake.

A fine 4 x 4 lens produces a very large amplitude if it lands on the anomaly.
Raw predictive variance therefore prefers asking a sequence of high-amplitude
one-vs-seven questions, even when a weaker 16 x 16 pulse would divide the
hypothesis set four-vs-four.

In other words:

> **largest predicted scalar spread is not the same as most informative
> question.**

That is the same family of mistake Gate 3 exposed when raw variance chased a
noisy channel.

## Gate 6B — post-result information policy

After the failure, a new selector was added explicitly as a post-result audit.

It scores each candidate lens by expected Bayesian information gain,
`I(hidden identity ; next noisy pulse)`, rather than raw amplitude variance.

Nothing else changes.

Result:

```text
raw variance multiscale       4.375 onset probes
fine-only                     4.375
expected information          3.000

information / fine            0.686
information / raw variance    0.686
localization accuracy         1.000
```

And the selected scale sequence was not merely "sometimes coarse":

```text
16 -> 8 -> 4
```

on **160 / 160** audited onset searches.

Classification:

`INFORMATION_GAIN_UNLOCKS_COARSE_TO_FINE_SEARCH_POSTHOC`

This does **not** retroactively pass the original Gate-6 multiscale
preregistration. It establishes a new mechanism after the negative result.

## What is interesting here

The old PerceptionLab workflow now has three causally distinct pieces.

### 1. Surprise decides **when**

The cheap HOME pulse gates expensive sensing.

### 2. Information decides **where and at what scale**

A useful lens is not the lens with the biggest response. It is the lens whose
possible responses partition the current hypotheses most usefully.

### 3. Persistent write changes the **next** decision

Without WRITE, successful perception does not alter future evidence gathering.
The same surprise has to be solved again.

With WRITE:

```text
observation_t
    -> spatial state_{t+1}
    -> prediction_{t+1}
    -> surprise_{t+1}
    -> whether another observation is purchased
```

That is considerably closer to a computational architecture than the original
checkerboard accident.

## What this gate does not earn

Gate 6 still receives:

- the anomaly prototype dictionary;
- the known global transport;
- a hand-specified HOME channel;
- a finite candidate lens bank.

So it has not learned its own ontology or sensing apparatus.

The obvious next question is therefore not another neuron bridge.

It is:

> Can the system learn which recurring residuals deserve persistent prototypes,
> and can those learned writes themselves create a useful future sampling
> policy?

That would move from **known prototypes + active memory** toward a genuinely
self-organizing PerceptionLab.
