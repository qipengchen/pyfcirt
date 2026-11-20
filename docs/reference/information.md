# Information and reliability

(conditional-fisher-information)=

## Expected information

`information()` computes expected Fisher information for the recorded
measurement events at one parameter point. It holds calibration fixed and
keeps both trait axes, including cross-trait information.

(joint-tirt-full-ranking-information)=

Information for a joint ranking uses the probability of the recorded event.
For TIRT or GTUM pair factors, the result adds comparison information.
This is a working independence approximation for comparisons derived from
the same rank. For Gaussian rankings, check the numerical refinement outputs.

## Posterior reliability

`reliability()` uses variation of estimated scores across people and
their posterior uncertainty. Use `empirical_reliability()` when you
already have means and variances on labelled person axes.

(experimental-mixed-block-information)=

## Information-based reliability

`information_reliability()` uses the local covariance approximation
$[I(\theta)+R^{-1}]^{-1}$, where $I$ is likelihood information and $R$
is the fixed population covariance. The approximation requires a shared
answered form and one zero-mean normal population.

[Frick (2023), Fisher information for forced-choice blocks](https://doi.org/10.1007/s11336-023-09931-8).

## API entries

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   ~pyfcirt.information
   ~pyfcirt.reliability
   ~pyfcirt.empirical_reliability
   ~pyfcirt.information_reliability
```
