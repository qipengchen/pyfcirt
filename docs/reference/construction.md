# Model construction and estimation

Prepare your data in `Model` and set custom priors or covariates before estimating.
Use `fit()` for NUTS.
Use `attach_samples()` to restore samples into a compatible fresh model.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   ~pyfcirt.Model
   ~pyfcirt.Model.set_priors
   ~pyfcirt.Model.fit
   ~pyfcirt.Model.attach_samples
   ~pyfcirt.set_platform
```

## Native and experimental estimation

Use `build()` to access the native PyMC graph. Posterior approximation supports
experimental recursive scoring. `sample_gtum()` provides specialized GTUM
estimation.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   ~pyfcirt.Model.build
   ~pyfcirt.Model.posterior_approximation
   ~pyfcirt.sample_gtum
```

```{toctree}
:hidden:

../examples/gpu
```
