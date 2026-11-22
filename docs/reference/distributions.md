# Native measurement distributions

Import the native measurement distributions from `pyfcirt.models.dist`.
Use these classes inside a PyMC model you define.

```python
from pyfcirt.models.dist import TIRTDistribution, TIRTRankDistribution
```
Call a class with a name and `observed=` to register an observed variable.
Use `.dist()` for an unnamed distribution and `pm.logp()` to evaluate it
at a response. Each page shows both forms.

Give the distributions arrays with known shapes. The last axes are described
on each API page and leading observation axes broadcast where applicable.
In a native graph, use zero-based category indices for categorical responses and
the actual continuous rating for Beta responses. Define priors and
identification constraints in your own model.
TIRT and GTUM Beta responses require positive precision for both density
evaluation and random draws. PyMC checks the derived Beta shapes in the log density.

## Joint response models

For a joint choice and response-time model, align both measurements on the
same observation axis. You can evaluate the sum of their native
`pm.logp()` expressions with `pm.compute_deterministics(..., backend="jax")`
to obtain their joint log likelihood per observation. The
[JRT–TIRT example](../tutorials/jrt_tirt_native.ipynb) shows the full analysis.

## Distribution reference

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   ~pyfcirt.models.thurstonian.tirt.TIRTDistribution
   ~pyfcirt.models.thurstonian.gtum.GTUMDistribution
   ~pyfcirt.models.thurstonian.rank.TIRTRankDistribution
   ~pyfcirt.models.sequential.TwoPLRankDistribution
   ~pyfcirt.models.sequential.GGUMRankDistribution
   ~pyfcirt.models.mgppm.MGPPMDistribution
   ~pyfcirt.models.likert.DichotomousIRTDistribution
   ~pyfcirt.models.likert.GPCMDistribution
   ~pyfcirt.models.likert.GGUMDistribution
```

Use [PairwiseL1](dif.md#pairwisel1) as a prior on zero-sum group contrasts
in a native PyMC graph.
