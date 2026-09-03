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
