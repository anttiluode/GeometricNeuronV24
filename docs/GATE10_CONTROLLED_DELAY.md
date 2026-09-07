# Gate 10 — Takens-style history and efference copy finally separate cleanly

Gate 9R established that a short history of correctly addressed scalar returns
can reconstruct a stationary hidden target once the adaptive-policy shortcut is
removed.  But a stationary inverse problem is still not the right place to make
a strong delay-embedding analogy.

Gate 10 makes the hidden state move.

## Controlled hidden oscillator

Each episode has hidden state

```text
theta_t    phase on a circle
omega      unknown constant angular drift
```

The observer issues one of four meaningless actions.  Action `a_t` both applies
a known phase kick and selects one scalar sensor direction:

```text
theta_t = theta_(t-1) + omega + kappa[a_t]
y_t     = cos(theta_t - phi[a_t]) + noise
```

The observer never sees `theta` or `omega`.  It knows only its own recent actions
and the scalar returns they produced.

For every candidate `(theta_t, omega)`, the estimator replays the known action
history backward through the delay window.  The candidate that best explains
the returns predicts the next scalar return, conditioned on a copy of the next
outgoing action.

This is a deliberately explicit **forced / input-output delay reconstruction**.
It is much closer to the Takens/efference intuition than the earlier stationary
gates, while remaining a synthetic mathematical toy.

## History length result

Across 8 clean trajectories:

| delay window | next-return MSE | omega MAE |
|---:|---:|---:|
| 1 | 0.51485 | 0.06509 |
| 2 | 0.02416 | 0.03159 |
| 3 | 0.00572 | 0.01823 |
| 4 | 0.00211 | 0.01119 |
| 6 | 0.000684 | 0.00577 |
| **8** | **0.000527** | **0.00371** |

This time there is no ambiguity about whether history is doing useful work.  A
single current observation leaves the hidden phase/drift badly underdetermined.
Adding action-conditioned history progressively reconstructs the moving state and
its hidden velocity.

## Efference / address attackers

At the best tested window (`K=8`):

```text
correct action history + known next action   MSE 0.000527
wrong labels on past action history          MSE 1.00017
remove copy of the next outgoing action      MSE 0.50446
```

Two distinct pieces are therefore required in this toy:

```text
past action + return history   -> reconstruct what state I am in
next action copy               -> predict what my own next act will cause
```

That is the clean mathematical bridge:

```text
DELAY EMBEDDING / STATE RECONSTRUCTION
        history -> hidden-state estimate

EFFERENCE COPY / KNOWN INPUT
        own action -> predicted self-generated transition/return

INNOVATION
        actual return - predicted self-return
```

They are complementary operations, not the same equation.

## Electric-fish-style innovation test

Occasional external phase jumps were inserted without reporting them to the
observer.

There were 282 shocks and 5,150 ordinary transitions.

```text
AUC using raw |new sensory value - old value|      0.520
AUC using |actual - predicted self-return|          0.934
AUC with wrong past action labels                   0.445
```

Raw sensory change is almost useless because the animal/system's own random
actions legitimately create large sensory changes.  Once the internal model
accounts for those actions and the reconstructed hidden state, an unreported
external perturbation becomes obvious in the residual.

That is the closest result in this branch to the mormyrid negative-image logic:

> **predict the sensory consequence of the known self-action from a state
> reconstructed out of recent action/return history; subtract that prediction;
> the unexpected world appears in the remainder.**

## What this says about the old Geometric Neuron intuition

The useful GeometricNeuron object was always

```text
address + scalar pulse + history
```

rather than a naked scalar pulse.  Gate 10 now shows why that object can matter
in a genuinely dynamical setting: the action/address history supplies the
coordinates needed to infer a hidden moving state, after which the next outgoing
action can be propagated through a forward model.

A cable/tree that naturally creates many delayed or differently filtered copies
of activity could, in principle, provide a temporal basis for such an estimator.
That is an engineering/biological hypothesis worth testing.  It is **not** yet a
claim that a dendrite implements Takens' theorem.

## AIS grating remains killed

Nothing in Gate 10 makes the ~190 nm actin/spectrin periodic scaffold into the
needed delay bank.  The relevant state history here spans successive behavioral /
computational events, not sub-microsecond propagation between cytoskeletal rings.

AIS remains interesting as the place where an integrated private state is turned
into an outgoing travelling event and where excitability can be contextually
controlled.  The delay/reconstruction machinery belongs elsewhere unless biology
provides evidence connecting them.

## Next biological/mechanistic question

The mathematical object is now clear enough to ask a better biological question:

> **Can realistic cable / synaptic / recurrent temporal filters approximate the
> forced-delay state estimator without an explicit digital action-history list?**

That is different from asking whether a dendrite is "Takens."  We can feed known
outgoing events into a physically parameterized filter bank, let path lengths,
membrane constants, synaptic kinetics and recurrent traces create the basis, and
measure whether a local downstream decoder can recover the same innovation
signal.

Frozen receipt: [`results/gate10/gate10_summary.json`](../results/gate10/gate10_summary.json).
