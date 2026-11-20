# Choose a backend and estimation method

## The posterior each method targets

Let $\mathbf y$ be the recorded answers, $\boldsymbol\theta$ the trait
scores, and $\boldsymbol\phi$ the remaining model parameters. Calibration
combines the [TIRT response probabilities](../tutorials/tirt_workflow.ipynb#model-the-comparisons)
with the priors:

$$
p(\boldsymbol\theta,\boldsymbol\phi\mid\mathbf y)
\propto
p(\mathbf y\mid\boldsymbol\theta,\boldsymbol\phi)\,
p(\boldsymbol\theta,\boldsymbol\phi).
$$

The first term on the right is the likelihood of the recorded answers.
For new respondents, this example fixes $\boldsymbol\phi$ at the calibration
posterior means and updates just the new trait scores using their answers.

NUTS and SMC produce draws or particles from the posterior target.
Variational inference fits an approximate distribution. MAP finds a mode
$\hat{\boldsymbol\theta}$ of the scoring posterior. A Laplace approximation
uses the mode as its center and covariance $\mathbf H^{-1}$, where
$\mathbf H$ is the Hessian of the **negative** log posterior at that mode.
The square roots of its covariance diagonal give the approximate score SDs.

Use `Model.fit()` for NUTS calibration and `Model.score()` for NUTS scoring.
Use `Model.build()` when you want to run a native PyMC estimator and inspect
its results directly.

```{image} ../_static/diagrams/estimation-methods.svg
:alt: Four branches of a prepared TIRT model show the outputs from NUTS, MAP with Laplace uncertainty, variational inference and SMC. The diagram identifies separate controls for the NUTS sampler and graph compilation.
```

## Calibrate with NUTS

Start with the data and model declaration from the [TIRT example](../tutorials/tirt_workflow.ipynb).

```python
import pyfcirt as fc

example = fc.load_example("hexaco_fc3", model="TIRT", n_persons=120, random_seed=42)
calibration = fc.Model()
calibration.add_fc(
    example.data.design,
    example.data.responses.frame,
    format="block_long",
    response_format="mole",
    model="TIRT",
)
calibration.prepare()
```

Choose one fit. BlackJAX is the default. Pass `nuts_sampler="pymc"` for
native PyMC NUTS or `nuts_sampler="numpyro"` for NumPyro. NumPyro and nutpie
are optional dependencies. The two arguments serve different purposes:
`nuts_sampler` selects the sampler, and `backend` selects graph compilation.

```python
calibration.fit(draws=1200, tune=1800, chains=4, random_seed=43)
```

For nutpie with JAX gradients, use this fit call in place of the preceding one.

```python
calibration.fit(
    nuts_sampler="nutpie",
    backend="jax",
    compile_kwargs={"gradient_backend": "jax"},
    random_seed=43,
)
```

With no draw settings, PyFCIRT uses 1,000 warmup iterations and 1,000
retained draws per chain, with target acceptance 0.8. PyMC chooses the chain
count. For BlackJAX and NumPyro, the package selects parallel chains on CPU
and vectorized chains on GPU. [PyMC NUTS options](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.sample.html) · [GPU setup](gpu.md)

## Prepare held-out respondents

Select three people whose responses were absent from calibration. In your
own study, this step uses the new response table with the same block and
statement IDs as the calibration design.

```python
source = fc.load_example("hexaco_fc3", model="TIRT")
responses = source.data.responses.frame
seen = example.data.responses.frame.person_id
held_out = responses.loc[~responses.person_id.isin(seen)]
people = held_out.person_id.drop_duplicates().head(3)
new = fc.ResponseTable(held_out.loc[held_out.person_id.isin(people)], format="block_long")
```

## Score with NUTS

Choose the scoring sampler in the same way as for calibration. This call
conditions on the calibrated parameter means and estimates the new trait
scores. The results are on `scored.idata`.

```python
scored = calibration.score(new, nuts_sampler="numpyro", backend="jax", random_seed=45)
```

## Score with MAP and Laplace uncertainty

For native estimation, create the same scoring target and build its PyMC
model. This target has one free variable, the centered trait array
`theta(person, trait)`. [PyMC find_MAP](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.find_MAP.html)

```python
import pymc as pm

target = calibration.score_model(new)
native = target.build()
with native:
    result = pm.find_MAP(
        compute_hessian=True,
        return_inferencedata=True,
        jitter=True,
        random_seed=46,
        backend="jax",
        progressbar=False,
    )
```

Check `result["optimizer_result"]["success"]` and the curvature at the
optimum before interpreting the approximation. For this centered scoring
target, the inverse curvature is returned as a covariance matrix on the trait
scores. Select its rows in person–trait order. The square roots of the
diagonal entries give local approximate SDs.

```python
import jax.numpy as jnp
import xarray as xr

theta = result.posterior.theta.isel(chain=0, draw=0, drop=True)
theta = theta.transpose("person", "trait")
labels = [
    f"theta[{person},{trait}]" for person in theta.person.values for trait in theta.trait.values
]
cov = result.fit.covariance_matrix.sel(rows=labels, columns=labels)
variance = jnp.diag(jnp.asarray(cov.values)).reshape(theta.shape)
sd = theta.copy(data=jnp.sqrt(variance))
scores = xr.Dataset({"MAP": theta, "Laplace SD": sd})
print(scores.sel(trait="honestyhumility").to_dataframe().drop(columns="trait").round(2))
```

```text
         MAP  Laplace SD
person
2       0.50        0.41
4       0.33        0.39
8      -0.34        0.41
```

The table shows Honesty–Humility scores at the posterior mode. The SD measures
local uncertainty around that mode, conditional on the fixed calibration.

## Use variational inference or SMC

Run either native estimator with the same `native` model. For calibration
instead of scoring, build an independent calibration model and use that
PyMC context.

```python
with native:
    approx = pm.fit(n=20000, method="fullrank_advi", backend="jax", random_seed=47)
    vi = approx.sample(draws=1000, random_seed=48)
```

Full-rank ADVI estimates a joint Gaussian approximation in PyMC's internal
coordinates. The sampled `vi.posterior.theta` gives trait scores on their
reported scale. [PyMC variational methods](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.fit.html)

```python
with native:
    smc = pm.sample_smc(draws=2000, chains=4, backend="jax", random_seed=49)
```

SMC gives a set of posterior particles. The ordinary TIRT model shown here
uses generative priors suitable for this approach. [PyMC SMC options](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.smc.sample_smc.html)

## Reuse results for information and reliability

For VI or SMC, use the mean and variance of the final `theta` samples for
[empirical reliability](../reference/generated/pyfcirt.empirical_reliability.md).
For MAP, use the mode and the Laplace variance calculated above.
For [information](../reference/generated/pyfcirt.information.md), pass a final
parameter point. A scoring point combines the estimated scores with the
model's fixed calibration parameters.
