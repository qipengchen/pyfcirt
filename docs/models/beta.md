---
orphan: true
---

# Experimental continuous comparisons

TIRT and GTUM with Beta links model **recorded continuous pair ratings** strictly
between zero and one. They are separate experimental mean models, with different
observations and parameterizations from categorical TIRT and GTUM. They do not
convert rankings, PICK/MOLE selections or ordinal categories into continuous data.
Declare `response_format=None` and `UnitIntervalScale` explicitly; a Beta link does not infer
the response format or response scale.

For each answered comparison,

```{math}
Y_{pij}\sim\operatorname{Beta}(\kappa\mu_{pij},\kappa(1-\mu_{pij})),
\qquad \operatorname{Var}(Y_{pij}\mid\mu_{pij},\kappa)
=\frac{\mu_{pij}(1-\mu_{pij})}{\kappa+1}.
```

There is one positive precision parameter per model. Its proper prior must be
specified explicitly; PyFCIRT does not select a default precision prior.
The public prior role is `response_precision`:

```python
import pymc as pm
import pyfcirt as fc

spec = fc.TIRT(
    link="beta_logit",
    priors={"response_precision": pm.Gamma.dist(alpha=2, beta=0.1)},
)
block = fc.BlockSpec("ratings", model=spec, response_format=None, scale=fc.UnitIntervalScale())
```

GTUM Beta uses the same prior key with `fc.GTUM(link="beta_logit", priors=...)`.
The example Gamma has mean 20; choose a proper prior for the rating dispersion
appropriate to your application.

Abilities have a centered multivariate normal prior with unit marginal variances.
The mean-link scale is fixed at one. There are no additional statement offsets,
estimated residual SDs or person/block effects in this baseline.

With positive loading magnitudes, statement keying signs {math}`s_j` and a canonical
statement-1 pair intercept {math}`\gamma_{ij}`, TIRT Beta uses

```{math}
z_{pij}=s_i\lambda_i\theta_{p,t(i)}
       -s_j\lambda_j\theta_{p,t(j)}-\gamma_{ij}.
```

Choose `TIRT(link="beta_probit")` or `TIRT(link="beta_logit")` explicitly. The mean is
{math}`\Phi(hz)` or {math}`\operatorname{logit}^{-1}(hz)`, respectively, where {math}`h=1`
when higher ratings prefer statement 1 and {math}`h=-1` otherwise. The logit is
unscaled. Direction applies to the whole contrast, including the intercept.

GTUM Beta uses an unscaled logit mean with

```{math}
z_{pij}=-\lambda_i|\theta_{p,t(i)}-\delta_i|
       +\lambda_j|\theta_{p,t(j)}-\delta_j|-\gamma_{ij}.
```

Location sides must be declared in the design, as for GTUM. Item loading and
location prior defaults reuse the corresponding categorical families. Comparison
intercepts are unordered scalar offsets, not category cutpoints. Reused statements
share one loading/location vector across blocks; comparison intercepts
remain specific to their blocks. An appropriate design is still needed for
identification.
