# Method

## Finite edge-block spectral triple

For a tree `G=(V,E)`, use `A=C(V)` and `H=direct_sum_(e in E) C^2`. For observable `a:V->C`, represent it on edge `e=(u,v)` as `pi(a)|e=diag(a_u,a_v)`. The edge Dirac block is `D_e=[[0,g_e],[g_e,0]]`, `g_e>0`.

The commutator block has operator norm `g_e |a_u-a_v|`, so

```text
||[D,pi(a)]|| = max_e g_e |a_u-a_v|.
```

Hence the Connes Lipschitz ball imposes `|a_u-a_v| <= 1/g_e` edgewise. On a tree, summing those bounds along the unique path gives an upper bound on `|a_i-a_j|`; the weighted-distance-from-i vertex function saturates it. Therefore

```text
d_D(i,j)=sum_(e on path i->j) 1/g_e.
```

This exact equality is specific to this deliberately chosen direct-edge construction. Other graph Dirac operators need not reproduce weighted shortest-path distance.

## Resource-constrained metric objective

Gate 1 uses a depth-4 complete binary tree: 31 vertices and 30 edges. Eight source/target pairs are training traffic and eight different combinations are held out. The split has identical expected edge-flow vectors by construction; it isolates shared-corridor geometry and is not a broad pair-generalization claim.

Let `F_e` be the probability that task traffic crosses edge `e`. Then expected spectral task distance is `sum_e F_e/g_e`. Add a standing-body tax `lambda sum_e 1/g_e` and enforce `sum_e g_e=B`:

```text
J(g)=sum_e (F_e+lambda)/g_e.
```

The Lagrange condition gives the exact budget oracle

```text
g*_e = B sqrt(F_e+lambda) / sum_j sqrt(F_j+lambda).
```

For one sampled task, `q_e` is 1 on its path and 0 otherwise. The stochastic negative gradient is `(q_e+lambda)/g_e^2`. Subtracting its mean projects the update onto the fixed-sum budget hyperplane:

```text
g_e <- g_e + eta[(q_e+lambda)/g_e^2 - mean_j((q_j+lambda)/g_j^2)].
```

The traffic bit and coupling are local to the edge; the mean subtraction is the global resource homeostat.

## Attackers

**Frozen:** all `g_e=1`.

**Traffic proportional:** `g_e proportional to F_e+lambda`, a parameter-free “strength tracks use” caricature.

**Shuffled traffic:** identical online update and identical amount of path traffic, but each traffic vector is randomly permuted across edge addresses before plasticity.

**Analytic oracle:** exact constrained minimum of the stated objective.