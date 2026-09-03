# Gate 4 — soma observability on real cell 1125

Gate 3 established a computational principle on a synthetic field: an addressed
write can reveal a hidden operator that passive reading cannot distinguish.

Gate 4 asks the first neuronal version of that question on the pinned released
human L2/3 morphology used by Operaattori.

## What is real here

The morphology is the released cell-1125 ASC:

`2013_03_06_cell11_1125_H41_06.asc`

from `ido4848/FCI`, pinned at commit
`75ad8b4d81a7f51bf888b30650c543592340db06`.

MorphIO parses **12,632 point-tree nodes** and Gate 4 finds **182 eligible
dendritic sections**.

## What is simplified here

This is **not** the released FCI NEURON model.

Gate 4 builds a passive DC cable directly from the MorphIO point tree with:

```text
Ra = 150 ohm cm
Rm = 20,000 ohm cm^2
```

Each morphology edge contributes axial conductance and membrane leak.  The only
readout is the synthetic soma-root voltage.

No NMDA, spikes, active channels, learning, or biological self-probing is used.

That restriction is deliberate: before adding nonlinear biology, ask whether
the tree itself plus addressed stimulation changes observability.

## Hidden mechanism

Twelve real dendritic sections distributed across path distance are chosen
deterministically.

One hidden parameter means:

> increase leak density on one selected section by 10%.

So there are twelve possible local operator changes.

## Probe family

A probe injects the same total current, **0.10 nA**, at a dendritic address.

The address can use one of three geodesic cluster radii:

```text
0 um      point probe
35 um     local cluster
110 um    broad dendritic cluster
```

There are 48 deterministic probe centers and therefore 144 candidate probes.

Every policy gets a budget of 12 soma measurements.

## Analytic sensitivity

For passive cable matrix `A`, probe current `b`, soma selector `e_s`, and
hidden local leak perturbation `D_p`:

```text
x = A^-1 b

dy/dp = - e_s^T A^-1 D_p A^-1 b
```

The complete probe × hidden-parameter matrix is therefore an observability
Jacobian.

The left solve is a **software adjoint**.  Gate 4 does not identify a
backpropagating action potential with this adjoint.

A direct 10% finite-difference perturbation checks the analytic derivative:
relative error was **0.0002**.

## The preregistered result failed — usefully

The strict gate demanded at least 90% exact identification of the hidden
section under **0.001 mV = 1 microvolt RMS** soma noise.

It did not pass.

```text
fixed one address, varying scale
    numerical rank                    2 / 12
    noise-visible rank                1 / 12
    identity accuracy                 0.372

active point probes
    numerical rank                   12 / 12
    noise-visible rank                7 / 12
    identity accuracy                 0.761

active multiscale probes
    numerical rank                   12 / 12
    noise-visible rank                7 / 12
    identity accuracy                 0.779

random multiscale
    median numerical rank             9 / 12
    median noise-visible rank         4 / 12
    mean identity accuracy            0.539  (post-result 32-subset audit)
```

The original `>= 0.90` condition passed **0/10** new noise seeds.

So the correct classification is:

`ADDRESS_OPENS_FULL_RANK_BUT_SOMA_NOISE_LIMITS_IDENTITY`

## What that means

There are two different notions of "observable."

### Algebraically observable

With a fixed stimulation address, many hidden leak directions collapse onto
the same soma response.

Moving the stimulation address opens the Jacobian to full numerical rank
**12/12**.

That is a real mechanism-level result inside this passive model.

### Reliably readable through a noisy soma channel

Full rank is not enough.

At 1 microvolt RMS observation noise, only **7/12 singular directions** exceed
the one-sigma visibility ruler, and exact hidden-section identity is about 78%.

The bottleneck has moved:

```text
before addressed stimulation:
    missing independent directions

after addressed stimulation:
    directions exist, but several are tiny at the soma
```

This is more informative than a clean 100% pass.

## Post-result noise boundary

The noise sweep was added only after the 1-uV gate failed.

```text
soma noise RMS      active exact identity

0.10 uV                  1.000
0.25 uV                  0.977
0.50 uV                  0.905
1.00 uV                  0.777
2.00 uV                  0.543
5.00 uV                  0.258
```

So in this model, the old 90% identity ruler crosses around **0.5 uV RMS**.

That is a measured scale for this assay, not a claim about biological somatic
noise in vivo.

## Multiscale did not earn necessity

Active point-only probing already reached full rank and 0.761 identity
accuracy.

Allowing cluster scale to vary improved that only modestly to 0.779.  The
chosen 12-probe policy used:

```text
0 um        3 probes
35 um       2 probes
110 um      7 probes
```

So Gate 4 earns **address selection**, not a claim that multiscale stimulation
is necessary.

## Radius heterogeneity did not rescue the story

As a secondary attacker, every dendritic radius was replaced by the real-cell
median before rebuilding the cable.

The active flattened-radius model still had full rank and its smallest singular
value was actually larger:

```text
real radii       s_min = 6.398e-05 mV
flat radii       s_min = 1.308e-04 mV
```

Therefore this assay does not support the idea that the biological radius
profile is what creates the observability gain.

The gain survives a much more boring cable.

## What this teaches us about the neuron question

Within a passive morphology model:

> **the soma does not need to contain a literal reconstruction of the dendritic
> tree for addressed perturbations to expose hidden local parameters.**

But:

> **a soma-only scalar remains a severe amplitude bottleneck even after the
> algebraic ambiguity is removed.**

This makes the branch-local-state + global-return-signal idea more concrete.

A branch can know its own address/local state.  A global somatic signal can
carry consequence information.  Gate 4 says the global signal may contain the
needed directions across an experiment, but some directions are too attenuated
to be robustly decoded from that scalar alone.

That is not evidence that neurons perform tomography on themselves.  It is a
constraint on any such mechanism.

## Next honest experiment

The immediate next question is no longer "can address help?"

It can.

The next question is:

> **does the already-established local NMDA feedback in Operaattori amplify,
> rotate, or selectively rescue the weak soma-observability directions?**

That test must keep the passive transport, probe addresses and hidden
perturbations controlled, then add the released HUMAN AMPA/NMDA local law and
compare the observability spectrum against AMPA-only / frozen-block attackers.

If nonlinear closure does not rescue the seven weak directions, we should stop
trying to make soma-only readout into a complete dendritic monitor.
