# Morgue

### “Any stronger graph edge means smaller Connes distance”

Too broad. It is exact here because the finite spectral triple is deliberately built from independent two-point edge blocks. Other graph Dirac operators can give different Connes-distance functionals.

### “Learning changes spectral geometry, therefore this is new mathematics”

No. Spectral triples and graph spectral distance are established mathematics. This repo builds an explicit traffic-driven plasticity experiment around one finite construction.

### “Cells that fire together wire together is automatically the right geometric law”

No. Coupling directly proportional to traffic over-specializes under the fixed body budget: all-leaf distance goes `6.533 -> 19.467` and the regularized objective does not improve over frozen.

### “This is Ricci flow”

No. `D_t -> D_(t+1)` is metric deformation, but the update here is projected stochastic gradient descent on a finite objective. No Ricci curvature or Ricci-flow equation is established.

### “Biological synapses literally contract Connes distance”

Not established. The repo earns a finite mathematical model that could serve as a geometric-synapse abstraction. Biology would require a measurable mapping from synaptic or dendritic variables to an empirically useful `D` plus independent predictions.

### “Riemann / Hilbert–Pólya is now connected”

Not in this repo. Both use spectral-operator language, but OutoSynapsi currently contains no zeta operator, prime structure, or zero-spectrum claim.

### “Gate 1 proves the metric is computationally useful”

Not yet. Gate 1 optimizes an objective written directly in Connes distance. Gate 2 must compile the same couplings into independent local dynamics and test whether spectral distance predicts propagation or task performance better than boring graph metrics.

### “There is one universal geometric plasticity law”

No. Gate 2 gives an exact counterexample inside this repo. If the relevant cost is Connes path length `1/g`, the fixed-budget optimum is square-root in traffic. If the relevant transport cost is a Laplacian-like resistance `1/g²`, the optimum is cube-root in traffic. “Learning changes geometry” is incomplete until the physical/operator role of `g` is specified.


### “Connes distance is the signal travel time”

No. Gate 3 compiles the same couplings into an actual local continuous-time diffusion with edge rate `g²`. Across 23,040 propagation problems, Connes distance reaches held-out log-R² `0.738`, while the exact one-way MFPT quantity is directional source-side-volume resistance:

```text
sum |S_e(source)| / g_e².
```

The Connes metric remains exact for the Gate-0 spectral triple. It is simply not the dynamical travel-time metric of this different operator.

### “Resistance distance fully predicts directed diffusion”

Not one-way hitting time. Plain `sum 1/g²` reaches held-out log-R² `0.782`. The missing factor is how much graph volume sits behind each edge relative to the source. Round-trip/commute quantities are closer to ordinary resistance; directed first passage is not.

### “Gate-2 cube-root flow allocation is automatically optimal for D² dynamics”

No. Gate 2 used ordinary edge flow plus a standing edge tax. Actual MFPT weights a traversed edge by source-side component volume. The resulting cube-root law acts on a different coefficient vector and reduces the measured dynamic objective from `69.147` to `47.844`.

### “Traffic proportional is a good practical shortcut”

Not under a whole-body objective. It makes favored task MFPT low (`50.456`) but explodes mean all-leaf MFPT to `1724.071`, worse than frozen by about 17×.
