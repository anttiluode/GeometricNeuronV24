# Gate 5 — NMDA does not rescue the weak soma-observability directions

Gate 4 reached the pinned human L2/3 cell-1125 morphology and found a useful
split:

- moving the dendritic stimulation address opens the passive 12-parameter
  soma-observability Jacobian from fixed-address rank 2 to full rank 12;
- but at the locked 1-uV RMS soma-noise ruler only 7/12 singular directions are
  practically visible.

Gate 5 asked whether the HUMAN voltage-dependent NMDA law already established
in Operaattori could bridge that remaining gap.

## Locked comparison

Every condition used the **same 12 probes selected by the passive Gate-4
policy**. No condition was allowed to choose friendlier addresses.

The hidden parameters, morphology, soma-only readout and 0.10-nA resting input
were also held fixed.

The four input laws were:

1. **PASSIVE_CURRENT** — the Gate-4 fixed current probe.
2. **AMPA_ONLY** — a conductance input rest-matched to 0.10 nA.
3. **FROZEN_NMDA** — the HUMAN AMPA/NMDA raw conductance ratio with magnesium
   block frozen at -70 mV, also rest-matched.
4. **HUMAN_NMDA** — same raw ratio and same resting current, but with
   voltage-dependent Jahr-Stevens block, gamma = 0.078 /mV.

In this static assay AMPA_ONLY and FROZEN_NMDA are deliberately algebraically
identical after rest matching. Their measured discrepancy was
**8.674e-19 mV**, providing an exact control.

## Result

```text
condition             rank   visible @1uV   s_min (mV)     identity

PASSIVE_CURRENT       12/12      7/12       6.3980e-05      0.776
AMPA_ONLY             12/12      7/12       5.2641e-05      0.713
FROZEN_NMDA           12/12      7/12       5.2641e-05      0.713
HUMAN_NMDA            12/12      7/12       5.4218e-05      0.724
```

Relative to the frozen-block control:

```text
noise-visible rank gain                    +0
exact identity gain                      +0.011
median gain on Gate-4 weak directions     1.030x
median gain on Gate-4 strong directions   1.036x
weak / strong selectivity                 0.994x
```

The HUMAN implicit derivative was checked against a direct 10% hidden-leak
finite difference; relative error was **0.0002**.

Classification:

> **HUMAN_NMDA_RESHAPES_BUT_DOES_NOT_RESCUE_SOMA_OBSERVABILITY**

## Why this is a real negative

The gate was not asking whether NMDA can produce nonlinear dendritic effects.
Operaattori already established that it can.

The gate asked a narrower question:

> Does HUMAN magnesium feedback preferentially amplify the five weak hidden
> directions that passive transport leaves buried at the soma?

It does not in this assay.

The weak directions gain about 3.0%. The strong directions gain about 3.6%.
So the effect is almost perfectly non-selective, and the number of
noise-visible directions remains 7/12.

The exact-identification gain from 0.713 to 0.724 is real but far below the
locked +0.05 rescue criterion.

## One additional lesson

Replacing a fixed current probe with an ordinary rest-matched conductance
already *reduces* soma identity from 0.776 to 0.713.

HUMAN NMDA recovers only a small fraction of that loss.

So voltage-dependent synaptic input is not automatically an observability
amplifier. Local nonlinearity can be computationally important while remaining
poor at exporting detailed local state to a distant scalar readout.

## What this does — and does not — say about neurons

This does **not** prove that biological neurons cannot make local state
available to the soma. Gate 5 remains a reduced static morphology-graph model.
It does not contain the complete released NEURON cell, active dendritic
channels, spikes, timing codes, calcium, inhibition, plasticity, or a biological
probe policy.

It does close the specific bridge we were attempting:

```text
real morphology
    +
addressed dendritic stimulation
    +
HUMAN NMDA feedback
    ->
high-fidelity soma reconstruction of local dendritic state
```

That route did not work.

The more conservative computational picture is therefore strengthened:

```text
branch-local state / branch-local nonlinearity
                 |
                 | local computation and plasticity
                 v
            branch consequence
                 |
                 v
       strongly compressed transport
                 |
                 v
          soma / axon broadcast
```

The soma can coordinate consequences without analytically reconstructing every
branch state.

That is a model interpretation, not a proof that all neuronal computation
*must* be decentralized.

## Project consequence

This is a good stopping point for the neuron bridge.

GeometricNeuronV24 should now return to the stronger synthetic question that
started in the PerceptionLab accident:

> What can an addressable spatial read/write system do when it remembers,
> predicts, measures surprise, moves its lens, and pays only when uncertainty
> makes another observation worthwhile?

The neuron work remains valuable as a boundary condition: a severe scalar
readout is not a natural place to demand complete local-state reconstruction.

The next gate therefore returns to **PerceptionLab mode**, not to another
biological rescue mechanism.
