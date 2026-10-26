# API reference

This page is generated from the installed package's docstrings. Examples and
statistical explanations are maintained in the guides; API signatures and method
descriptions follow the source code.

## Data and model construction

```{eval-rst}
.. autoclass:: pyfcirt.FCDesign
   :members:

.. autoclass:: pyfcirt.FCData
   :members:

.. autoclass:: pyfcirt.Model
   :members:

.. autoclass:: pyfcirt.Prior
   :members:
```

## Fit results

```{eval-rst}
.. autoclass:: pyfcirt.results.FitResult
   :members:
```

## Scoring and persistence

```{eval-rst}
.. autofunction:: pyfcirt.scoring.calibration_parameters

.. autofunction:: pyfcirt.scoring.score

.. autoclass:: pyfcirt.scoring.ScoreResult
   :members:

.. autofunction:: pyfcirt.io.save

.. autofunction:: pyfcirt.io.load

.. autoclass:: pyfcirt.io.SavedFit
   :members:
```

## Simulation

```{eval-rst}
.. autofunction:: pyfcirt.simulation.simulate_mgppm
```
