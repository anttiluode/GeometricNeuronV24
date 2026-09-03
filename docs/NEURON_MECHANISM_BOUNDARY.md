# Neuron-mechanism boundary after Gate 3

Gate 3 creates a computational distinction that maps cleanly onto two different kinds of neuronal experiment. It does **not** show that a biological neuron implements the V24 address bus.

## The distinction earned in V24

V24 now has two write primitives:

```text
STATE WRITE
change x
leave operator G fixed
observe how G transforms the perturbation

OPERATOR WRITE
change a local parameter of G
erase x
load a common diagnostic state
observe the changed future response
```

Gate 3A establishes the first in a controlled transport system. Gate 3B establishes the second in a controlled persistent-parameter system.

This matters for biological translation because stimulation and plasticity should not be conflated.

## Mechanistic translation to a dendrite

A future dendritic assay can use the same logic without pretending the image lattice is a neuron.

| V24 object | dendritic experiment analogue |
|---|---|
| hidden operator | cable + channel + synaptic parameters on a known arbor |
| read address | which dendritic site or cluster is stimulated |
| lens scale | single synapse, local cluster, branch, or larger recruited domain |
| scalar read | somatic voltage/spike statistic, or another explicitly chosen port |
| state write | diagnostic current/synaptic stimulation that changes voltage but not lasting parameters |
| operator write | a lasting local conductance, synaptic, channel-density, or geometric perturbation |
| erase fast state | wait/reset voltage and transient gating before the diagnostic assay |
| active policy | choose the next stimulation address from remaining mechanistic uncertainty |

The important biological question is therefore not "does the neuron contain V24?"

It is:

> **Which hidden dendritic parameter directions are observable from a soma-only readout, and which become observable when stimulation address and integration scale can be actively varied?**

## Cell 1125 experiment

Operaattori already compiled the public human L2/3 morphology `2013_03_06_cell11_1125_H41_06.asc` into a real branching scaffold. V24's first neuronal bridge should reuse that object rather than invent another tree.

Choose a controlled hidden-parameter family, for example:

- increased local leak on one branch;
- altered diameter on one branch;
- changed NMDA conductance in one synaptic cluster;
- changed inhibitory conductance at one dendritic location.

For every permitted stimulation probe `a`, record a somatic response statistic `y_a`. Construct the sensitivity matrix

```text
J[a, p] = d y_a / d p
```

where columns are hidden dendritic parameters.

The singular spectrum of `J` is the observability spectrum. Its right singular vectors show combinations of dendritic parameters that a given probe family can or cannot distinguish.

Required comparisons:

1. fixed stimulation address;
2. multiple addressed stimulation sites;
3. multiple cluster sizes / input scales;
4. passive cable only;
5. nonlinear dendritic mechanisms enabled;
6. real morphology versus radius/order/geometry controls.

If moving the stimulation address adds independent singular directions, the result would be about the **observability of real dendritic mechanisms**, not about image reconstruction.

## Backpropagating action potentials are not adjoints

A backpropagating spike is a real biological return signal. V24 does not identify it with the mathematical adjoint.

Likewise, the Bösch-Gediz-Türeci physical-backpropagation theorem gives conditions under which a reciprocal physical system can generate an adjoint field. V24 has not shown those conditions hold for an active dendrite, and Gate 3 does not use a physical adjoint at all.

Keep three objects separate:

```text
diagnostic perturbation
biological return signal
mathematical adjoint
```

They may interact in future experiments. They are not synonyms.

## What Gate 3 suggests, and what it does not

Gate 3 supports a general mechanism-level idea:

> **low observability from a fixed output does not imply that hidden structure is unknowable; active, spatially addressed perturbations can expose operator differences that passive observation cannot.**

That principle is suitable for a neuronal experiment.

Gate 3 does **not** establish:

- that neurons actively choose stimulation addresses;
- that dendritic inhibition is literally a V24 lens;
- that somatic spikes encode the whole dendritic state;
- that a neuron performs tomography on itself;
- that biological plasticity follows an adjoint gradient;
- that the V24 image lattice is a neuron model.

The next claim should be earned on the real morphology.


## Gate 4 result: the real tree opens, then the soma bottleneck appears

Gate 4 now performs the first version of that assay on the pinned cell-1125
morphology, using a passive morphology-graph cable and twelve possible local
10% leak-density changes.

The fixed-address family has numerical rank **2/12**. Active addressed
stimulation reaches **12/12**.

That is the positive part.

The locked soma-noise test is the boundary: at **1 uV RMS** only **7/12**
singular directions clear the one-sigma ruler and exact hidden-section identity
is about **0.778**. The original >=0.90 target passed **0/10** new noise seeds.

So the biological translation becomes more precise:

> Moving the stimulation address can remove an algebraic ambiguity without
> making the soma a high-fidelity monitor of all dendritic parameters.

This is compatible with a branch-local-state + global-consequence architecture:
local branches need not infer themselves from the soma. A global somatic signal
may carry useful consequence information while remaining insufficient for
central reconstruction of the complete dendritic state.

That is still a model-level constraint, not evidence that a neuron actively
runs this identification procedure.


## Gate 5: the tested NMDA bridge does not rescue the soma channel

The next bridge was run rather than argued.

On the exact Gate-4 passive-selected addresses, rest-matched HUMAN NMDA changed
the observability spectrum only weakly:

```text
frozen-block identity       0.713
HUMAN identity              0.724
visible rank                7/12 -> 7/12

weak-direction gain         1.030x
strong-direction gain       1.036x
```

So the voltage-dependent magnesium block does not preferentially amplify the
hidden directions that passive transport suppresses at the soma.

The defensible conclusion is narrower than “neurons must be decentralized.”
This particular morphology + static cable + HUMAN NMDA bridge does not support
the soma-as-detailed-central-observer picture.

That makes a decentralized interpretation more attractive: local branch state
and local plasticity can do computation while somatic output serves as a
strongly compressed consequence/coordination signal. But biological neurons
have many mechanisms absent here, so Gate 5 is a stopping line for this bridge,
not a proof about all neurons.
