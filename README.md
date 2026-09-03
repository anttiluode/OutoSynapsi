# OutoSynapsi — when a synapse changes distance

**Schanpps for all. Sol thinking repo.**

This repo nails down one narrow version of an old Geometric Neuron idea:

> **If a coupling is part of the Dirac operator of a finite spectral triple, changing that coupling changes the metric. Plasticity can therefore be literal geometry change.**

That sentence is exact **inside the construction used here**. It is not a claim that biological synapses implement Connes geometry, that every graph Dirac operator has this metric, or that weighted graphs needed noncommutative geometry to exist.

The experiment then asks the non-tautological part:

> **Can local traffic plus a fixed global coupling budget deform that metric in the useful direction, and can we derive the plasticity law rather than naming ordinary Hebbian strengthening “geometry”?**

## Gate 0 — the exact finite object

Let a tree have vertices `V`, edges `E`, and positive edge couplings `g_e`.

Use the commutative algebra

```text
A = C(V)
```

acting on

```text
H = direct sum over edges of C^2
```

by

```text
pi(a)|_(u,v) = diag(a_u, a_v).
```

For each edge `e=(u,v)` choose the Dirac block

```text
D_e = [[0, g_e],
       [g_e, 0]].
```

Then exactly

```text
||[D, pi(a)]|| = max_e g_e |a_u - a_v|.
```

The Lipschitz-ball condition `||[D,pi(a)]|| <= 1` therefore imposes

```text
|a_u - a_v| <= 1/g_e
```

edge by edge. On a connected tree the Connes distance between pure vertex states is therefore

```text
d_D(i,j) = sum_(e on unique path i->j) 1/g_e.
```

For a two-point “synapse” this collapses to

```text
d_D(0,1) = 1/g.
```

So in this finite model, increasing a coupling really does pull its endpoints closer in the spectral metric.

Executed audit:

```text
max two-point distance error       0
max explicit commutator error      1.78e-15
max distance-witness error         0
```

Classification:

> `EDGE_COUPLING_IS_EXACT_INVERSE_SPECTRAL_LENGTH_IN_THIS_TRIPLE`

Receipt: [`results/GATE0.json`](results/GATE0.json)

## Gate 1 — traffic deforms the metric

The first useful question is not whether we can rename edge weights “geometry.” We obviously can in the construction above.

The useful question is whether **experience can reshape that geometry under a resource constraint**.

The assay uses a 31-node binary tree with 30 couplings and a fixed total coupling budget

```text
sum_e g_e = 30.
```

A repeated task asks signals to connect two leaf populations. Eight source/target pairs are shown during plasticity; eight different pairs are held out. The held-out pairs have the same edge-flow statistics by construction, so this gate tests whether a shared corridor is learned rather than whether pair identities are memorized.

We minimize

```text
J(g) = E_task[d_D(source,target)]
       + lambda * sum_e 1/g_e
```

with `lambda=0.05`.

The second term is a **standing-body tax**: unused branches are still part of the organism, so the learner is not allowed to collapse all coupling into the currently fashionable route without cost.

If `F_e` is expected task traffic through edge `e`, then

```text
J(g) = sum_e (F_e + lambda) / g_e.
```

Under the fixed budget, the exact optimum is

```text
g*_e = B sqrt(F_e + lambda) / sum_j sqrt(F_j + lambda).
```

That gives us an analytic oracle before the online learner is run.

### The online plasticity rule

For one task, let `q_e=1` on the currently used path and zero elsewhere. Stochastic descent gives the local pressure

```text
(q_e + lambda) / g_e^2.
```

A single global homeostat subtracts the mean update so the coupling budget stays fixed:

```text
g_e <- g_e + eta * [
    (q_e + lambda)/g_e^2
    - mean_j((q_j + lambda)/g_j^2)
]
```

The traffic term is local to the edge. The only global quantity is the resource-budget homeostat.

## Gate 1 result

40 independent traffic streams, 10,000 updates each:

| rule | held-out task distance ↓ | all-leaf distance | regularized objective ↓ |
|---|---:|---:|---:|
| frozen uniform geometry | 8.000 | **6.533** | 9.500 |
| traffic-proportional coupling | **4.469** | 19.467 | 9.500 |
| analytic metric oracle | 4.982 | 8.324 | **7.0835** |
| **online metric plasticity** | **4.9825** | 8.323 | **7.0841** |
| shuffled-traffic attacker | 8.016 | 6.537 | 9.5166 |

The learned coupling vector has mean cosine similarity **0.999962** to the analytic oracle.

Its regularized objective is only **1.000084×** the oracle, while held-out task distance falls to **62.28%** of the frozen geometry.

Shuffling the exact same amount of traffic across edges destroys the contraction: held-out distance returns to **1.002×** frozen.

Classification:

> `TRAFFIC_DRIVEN_RULE_DEFORMS_SPECTRAL_METRIC_TO_NEAR_ORACLE_UNDER_BUDGET`

Receipt: [`results/GATE1.json`](results/GATE1.json)

## The useful surprise: naive Hebbian strengthening is not the metric optimum

The parameter-free `traffic-proportional` attacker assigns coupling directly in proportion to `(F_e + lambda)`.

It makes the repeated task path even shorter than the metric oracle: `4.469` versus `4.982`.

But it does so by nearly tripling mean distance across the rest of the leaves:

```text
6.533 -> 19.467
```

and its regularized objective is no better than the original uniform tree.

So the result is not merely

> high traffic -> thick synapse.

Under a fixed body budget, the geometry-aware optimum is a **square-root allocation**, balancing frequently used corridors against preserving the rest of the body.

That is the first thing in this repo that I would actually carry back to the old “geometric synapse” dream.

## What this earns

Inside this explicit finite spectral triple:

```text
synaptic coupling g_e
        ↓
Dirac operator D(g)
        ↓
Connes metric d_D
        ↓
traffic-dependent plasticity changes g
        ↓
learning literally changes metric geometry
```

And the deformation can be optimized under a resource budget.

## What this does **not** earn

- Biological synapses are not shown to be spectral triples.
- This is not Ricci flow.
- This is not a Riemann / Hilbert–Pólya result.
- A different graph Dirac operator can give a different Connes distance.
- The exact metric construction is not claimed as new mathematics.
- Gate 1 still optimizes a metric objective. It has **not yet shown that Connes distance predicts an independent dynamical computation better than ordinary graph quantities**.

That last item is the next real gate.

## Gate 2 — D versus D² changes the learning law

Gate 1 exposed a danger in the phrase “the synapse changes geometry”: **which operator is actually doing the physics?**

If `g_e` is the inverse Dirac length in the Gate-0 triple, the path metric scales as

```text
Dirac / Connes cost ~ 1/g_e.
```

But a Laplacian-like transport built from squared couplings has edge conductance `g_e²`. On a tree its effective resistance scales as

```text
D² / Laplacian resistance cost ~ 1/g_e².
```

Under the same fixed coupling budget, those two objectives have different exact optima:

```text
Dirac metric objective:      g*_e ∝ (F_e + lambda)^(1/2)
D² resistance objective:     g*_e ∝ (F_e + lambda)^(1/3)
```

That difference is not cosmetic. Gate 2 runs a second online plasticity law, derived from the `1/g²` objective:

```text
g_e <- g_e + eta * [
    2(q_e + lambda)/g_e^3
    - mean_j(2(q_j + lambda)/g_j^3)
]
```

Across 40 seeds:

| geometry / rule | task resistance ↓ | all-leaf resistance | D² objective ↓ |
|---|---:|---:|---:|
| frozen | 8.000 | **6.533** | 9.500 |
| Gate-1 square-root allocation | **3.332** | 14.515 | 7.083 |
| **D² cube-root oracle** | 4.008 | **9.058** | **6.3257** |
| **online D² plasticity** | **4.012** | **9.048** | **6.3262** |
| traffic proportional | 3.255 | 111.230 | 31.863 |
| shuffled traffic | 8.017 | 6.537 | 9.518 |

The online D² rule reaches **1.000084×** the analytic cube-root oracle and coupling cosine **0.999986**.

Classification:

> `DIRAC_METRIC_AND_D_SQUARED_TRANSPORT_REQUIRE_DIFFERENT_BUDGET_OPTIMA`

This is the most important boundary so far:

> **“Learning changes geometry” is incomplete until we say which operator defines the geometry and which operator actually carries the signal.**

If biology or an artificial body uses a Dirac-like quantity for one function and a Laplacian / diffusion generator for another, the same traffic can imply different optimal structural changes.

Receipt: [`results/GATE2.json`](results/GATE2.json)

## Gate 3 — actual local diffusion chooses a different geometry

Gate 2 still optimized analytic path costs. Gate 3 finally compiles the same couplings into an **independent local dynamical system**.

Each tree edge carries symmetric continuous-time transition rate

```text
c_e = g_e^2
```

and the body evolves under the corresponding local generator. Starting from one source node, the measured propagation quantity is the **mean first-passage time (MFPT)** to a target. The propagation solve never sees Connes distance or any graph-distance formula.

Across 96 random fixed-budget coupling geometries × 240 ordered leaf pairs = **23,040 actual propagation problems**:

| proposed predictor | held-out log R² | Spearman |
|---|---:|---:|
| hop count | 0.331 | 0.455 |
| bottleneck inverse conductance | 0.632 | 0.766 |
| **Connes distance** `sum 1/g` | **0.738** | **0.822** |
| resistance distance `sum 1/g²` | 0.782 | 0.849 |
| **directional volume resistance** | **1.000** | **1.000** |

The exact winner is

```text
MFPT(s -> t)
  = sum_(e on path s->t) |S_e(s)| / g_e^2
```

where `|S_e(s)|` is the number of vertices on the **source side** after cutting edge `e`.

So the first actual-dynamics gate kills a seductive simplification:

> **Connes distance is a real metric of the Gate-0 spectral triple, but it is not the propagation time of this D²-like diffusion.**

Even plain resistance distance is incomplete for one-way transport because it ignores how much body sits behind each traversed edge. Direction and volume matter.

Classification:

> `LOCAL_DIFFUSION_REVEALS_DIRECTIONAL_VOLUME_GEOMETRY`

### Plasticity derived from the actual dynamics

This gives a new body-level objective:

```text
J_dyn(g)
  = mean held-out MFPT
    + beta * mean all-leaf MFPT
```

with `beta=0.05`.

If `K_e` is the expected directional source-side traffic coefficient, then

```text
J_dyn = sum_e K_e / g_e^2
```

and the exact fixed-budget optimum is again cube-root, but now of **directional volume traffic** rather than ordinary edge-use frequency:

```text
g*_e ∝ K_e^(1/3).
```

The online local rule samples task traffic plus background body traffic and updates only traversed edges, with one global fixed-budget homeostat.

Across 20 independent streams × 20,000 updates:

| allocation | held-out MFPT ↓ | all-leaf MFPT | dynamic objective ↓ |
|---|---:|---:|---:|
| frozen | 124.000 | **101.267** | 129.063 |
| Gate-1 Connes sqrt | 51.649 | 224.985 | 62.898 |
| Gate-2 resistance cube | 62.127 | 140.396 | 69.147 |
| traffic proportional | 50.456 | 1724.071 | 136.659 |
| **dynamic MFPT oracle** | **35.910** | 238.671 | **47.844** |
| **online dynamic plasticity** | **36.005** | **238.204** | **47.915** |

The online rule reaches **1.00150×** the analytic dynamic oracle with coupling-vector cosine **0.999903**.

The important correction is now sharper than “learning changes geometry”:

> **The body can support several legitimate geometries at once. The learning law depends on the actual local dynamics and on which notion of distance the task cares about.**

Receipt: [`results/GATE3.json`](results/GATE3.json) · [live Gate 3 diffusion demo](gate3.html)

## Next gate — can the body choose its own geometry?

Gate 3 still hands us the dynamical law `c_e=g_e²`. The next serious question is whether a local adaptive system can infer from consequences whether its useful geometry is closer to:

```text
1/g       Dirac / Connes
1/g²      resistance
|S|/g²    directional diffusion
or something else entirely
```

without being told the exponent or the correct path functional.

## Lineage

This repo is the precise version of a recurring old idea:

- [`Geometric-Neuron`](https://github.com/anttiluode/Geometric-Neuron) — geometry/history as computation.
- [`Operaattori`](https://github.com/anttiluode/Operaattori) — structure compiles an operator.
- [`GeometricNeuronV24`](https://github.com/anttiluode/GeometricNeuronV24) — bounded READ/WRITE interrogation.
- [`LentoOrava`](https://github.com/anttiluode/LentoOrava) — local facts can become distant global consequence; global scalar consequence can return action to local sites.

Here the claim is narrower:

> **a coupling can be an inverse length, and plasticity can be metric deformation.**

## Run

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python experiments/gate0_exact_metric.py
python experiments/gate1_metric_plasticity.py
python experiments/gate2_dirac_vs_laplacian.py
python experiments/gate3_actual_diffusion.py
```

## Mathematical references

- Alain Connes, *Noncommutative Geometry and Reality* — spectral distance recovers geodesic distance and applies to discrete spaces: https://alainconnes.org/wp-content/uploads/reality.pdf
- Alain Connes, *Noncommutative Geometry Year 2000* — `D` as inverse line element: https://alainconnes.org/wp-content/uploads/2000.pdf
- Manfred Requardt, *A New Approach to Functional Analysis on Graphs, the Connes-Spectral Triple and its Distance Function* — graph spectral triples and the warning that Connes distance depends on the chosen graph geometry / Dirac operator: https://arxiv.org/abs/hep-th/9708010

**Attackers first. Geometry second. Biology much later.**
