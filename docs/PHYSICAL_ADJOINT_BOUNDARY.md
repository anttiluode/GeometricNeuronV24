# The physical-adjoint paper: exact relevance and boundary

Reference:

> Cyrill Bösch, Yigithan Gediz, and Hakan Türeci. **Unifying Physical Backpropagation.** arXiv:2608.11585v1, 13 August 2026. [arXiv](https://arxiv.org/abs/2608.11585)

This is external work. Its theorem and claims are not results of GeometricNeuronV24.

## What V24 currently does

For the software measurement

```text
y = Hx
L = 1/2 ||H x_hat - y||^2,
```

V24 computes

```text
gradient_x L = H^T(H x_hat - y).
```

That transpose is the adjoint of the explicit software lens. Gate 0 verifies one directional derivative against a central finite difference.

This tells us where a candidate image change would affect the measured pulse mismatch. It does **not** show that a physical medium produced the adjoint field.

## What the paper asks

The paper asks when the same hardware used for the forward computation can, under changed drives/initial conditions and possibly reversed explicit time dependence, generate the adjoint field required for exact gradients.

Its organizing split is:

| regime | same-hardware statement in the paper |
|---|---|
| linear reciprocal or Onsager-reciprocal dynamics | a finite-amplitude adjoint run can be available; loss or gain is permitted when reciprocity is preserved |
| nonlinear trajectory dynamics | requires recoverable time reversal plus reciprocity of the linearized dynamics; ordinary overdamped nonlinear trajectories are obstructed on the same hardware |
| stationary linear problems | reciprocal fixed-point/Helmholtz cases can admit a single finite-amplitude experiment |
| stationary nonlinear problems | an infinitesimal free-versus-nudged construction is available under tangent self-adjointness; this recovers Equilibrium Propagation |

The paper also separates obtaining gradients from applying parameter updates. It assumes an external teacher for updates and does not solve autonomous structural development.

## Why it matters later

If V24 eventually replaces its explicit software lens with a reciprocal physical transport body, the paper supplies conditions to test before saying that the body can generate its own adjoint field.

The future comparison would need three distinct objects:

```text
analytic/software gradient
digital adjoint solve
same-device reciprocal experiment
```

They should first be compared in a linear reciprocal regime. Nonlinear transient dynamics cannot be waved through by pointing to an internal fixed-point solve. The loss and the learned object determine whether a stationary construction is actually applicable.

## What the paper does not give V24

- an image reconstruction algorithm;
- observability from a scalar port;
- an address-selection policy;
- a growth rule;
- a proof that local stress is a global biological objective gradient;
- a reason to call the ECG loop a physical backpropagator;
- evidence about a real neuron.

For now, the paper is a boundary marker and a future experimental recipe, not a result inside V24.
