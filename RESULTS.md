# Results ledger

## Gate 0 — `EDGE_COUPLING_IS_EXACT_INVERSE_SPECTRAL_LENGTH_IN_THIS_TRIPLE`

| audit | maximum absolute error |
|---|---:|
| two-point `d=1/g` | 0 |
| closed-form vs explicit matrix commutator norm | `1.776e-15` |
| distance witness vs path metric | 0 |

## Gate 1 — `TRAFFIC_DRIVEN_RULE_DEFORMS_SPECTRAL_METRIC_TO_NEAR_ORACLE_UNDER_BUDGET`

31-node tree, 30-edge coupling budget, `lambda=0.05`, 40 stochastic seeds, 10,000 updates per seed.

| rule | held-out task distance ↓ | all-leaf distance | objective ↓ |
|---|---:|---:|---:|
| frozen uniform | 8.000000 | 6.533333 | 9.500000 |
| traffic proportional | **4.468975** | 19.467003 | 9.500000 |
| analytic metric oracle | 4.981573 | 8.323659 | **7.083485** |
| **online metric plasticity** | **4.982478** | 8.323344 | **7.084082** |
| shuffled traffic | 8.016107 | 6.537045 | 9.516604 |

Online plasticity reached `1.000084×` the oracle objective, reduced held-out spectral distance to `0.622810×` frozen, and had mean coupling-vector cosine `0.999962` to the analytic oracle. Shuffled traffic returned held-out distance to `1.002013×` frozen.

The traffic-proportional rule is a useful negative: it shortens the favored corridor aggressively but expands average background leaf geometry from `6.533` to `19.467`, leaving the regularized objective at the same `9.5` as frozen.

## Gate 2 — `DIRAC_METRIC_AND_D_SQUARED_TRANSPORT_REQUIRE_DIFFERENT_BUDGET_OPTIMA`

The same edge coupling can enter two different operator-derived costs. Gate 0 Connes distance scales as `1/g`; a Laplacian-like `D²` conductance gives tree resistance scaling as `1/g²`.

| allocation | task resistance ↓ | all-leaf resistance | D² objective ↓ |
|---|---:|---:|---:|
| frozen | 8.000000 | 6.533333 | 9.500000 |
| Gate-1 square-root allocation | **3.332202** | 14.515181 | 7.083485 |
| D² cube-root oracle | 4.008201 | **9.057813** | **6.325698** |
| **online D² plasticity** | **4.012055** | **9.047936** | **6.326228** |
| traffic proportional | 3.255208 | 111.230414 | 31.863155 |
| shuffled traffic | 8.017374 | 6.536553 | 9.517874 |

The online D² rule reaches `1.000084×` the analytic cube-root oracle with coupling cosine `0.999986`.

The result kills the idea that one generic “geometry-aware plasticity” rule follows from traffic alone. The operator power matters.


## Gate 3 — `LOCAL_DIFFUSION_REVEALS_DIRECTIONAL_VOLUME_GEOMETRY`

The learned coupling is compiled into a symmetric continuous-time local generator with adjacent rate `c_e=g_e²`. Propagation is measured by source-to-target mean first-passage time.

### Predictor audit

96 random fixed-budget geometries × 240 ordered leaf pairs = 23,040 propagation problems.

| predictor | held-out log R² | MAPE | Spearman |
|---|---:|---:|---:|
| hop count | 0.3306 | 0.7763 | 0.4552 |
| bottleneck | 0.6322 | 0.6268 | 0.7660 |
| Connes distance `sum 1/g` | 0.7378 | 0.4171 | 0.8216 |
| resistance `sum 1/g²` | 0.7824 | 0.4220 | 0.8493 |
| **directional volume resistance** | **1.0000** | **3.66e-14** | **1.0000** |

Exact tree identity:

```text
MFPT(s->t) = sum_(e in path) |S_e(s)| / g_e²
```

where `|S_e(s)|` is source-side component size after deleting edge `e`.

### Dynamic plasticity

Objective: held-out MFPT + 0.05 × all-leaf MFPT.

| allocation | held-out MFPT ↓ | all-leaf MFPT | objective ↓ |
|---|---:|---:|---:|
| frozen | 124.000 | 101.267 | 129.063 |
| Gate-1 sqrt | 51.649 | 224.985 | 62.898 |
| Gate-2 cube | 62.127 | 140.396 | 69.147 |
| traffic proportional | 50.456 | 1724.071 | 136.659 |
| **dynamic oracle** | **35.910** | 238.671 | **47.844** |
| **online dynamic plasticity** | **36.005** | **238.204** | **47.915** |

Online objective/oracle = `1.001499`; mean coupling cosine to oracle = `0.999903`.


## Gate 4 — `SCALAR_CONSEQUENCES_IDENTIFY_EFFECTIVE_GEOMETRY`

Six hidden local generators use `c_e=g_e^p` with `p ∈ {0.75,1,1.5,2,2.5,3}`. The learner gets 16 noisy scalar source→target arrival measurements and fits only the family

```text
scale * sum |S_e(source)|^alpha / g_e^beta.
```

Across 120 worlds:

| quantity | result |
|---|---:|
| mean absolute alpha error | 0.01458 |
| mean absolute beta error | 0.00208 |
| mean held-out log R² | 0.999829 |
| mean held-out MAPE | 0.00551 |
| mean learned objective / oracle | **1.000136** |
| worst learned / oracle | **1.002638** |

Fixed Gate-3 `alpha=1,beta=2` averages `1.0711×` oracle and reaches `1.1880×` in the worst hidden world. Fixed Connes-like `alpha=0,beta=1` reaches `4.7196×` oracle at the worst hidden power.

Probe-budget sweep:

| pulses | mean / oracle | worst / oracle |
|---:|---:|---:|
| 4 | 1.005280 | 1.070568 |
| 8 | **1.000393** | **1.002638** |
| 16 | 1.000212 | 1.002223 |
| 32 | 1.000069 | 1.000516 |

Boundary: the two-parameter metric family is supplied; only its effective geometry is identified.
