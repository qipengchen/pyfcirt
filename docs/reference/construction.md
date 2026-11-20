# Model construction and estimation

Prepare your data in `Model` and set custom priors or covariates before estimating.
Use `fit()` for NUTS or `build()` for direct native PyMC access.
Use `attach_samples()` to restore samples into a compatible fresh model.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   ~pyfcirt.Model
   ~pyfcirt.Model.build
   ~pyfcirt.Model.set_priors
   ~pyfcirt.Model.fit
   ~pyfcirt.Model.attach_samples
   ~pyfcirt.Model.posterior_approximation
   ~pyfcirt.set_platform
   ~pyfcirt.sample_gtum
```

```{toctree}
:hidden:

../examples/gpu
```
