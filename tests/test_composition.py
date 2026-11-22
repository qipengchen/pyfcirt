"""Caller-owned tensors compose with native PyMC likelihoods without a facade."""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest
from numpy.testing import assert_allclose
from scipy.stats import norm

from pyfcirt.models.thurstonian.tirt import TIRTDistribution


@pytest.mark.parametrize("law", ["1PL", "2PL", "3PL", "4PL", "GPCM", "GGUM"])
def test_item_graph_prunes_zero_q_inputs_preserving_dense_values_and_gradients(law):
    from functools import partial

    import jax
    import jax.numpy as jnp
    import pandas as pd
    import xarray as xr

    import pyfcirt as fc
    from tests.helpers import labelled_point

    spec = fc.DichotomousIRT(kind=law) if law.endswith("PL") else getattr(fc, law)()
    source = fc.ItemInput(
        pd.DataFrame(
            dict(
                item_id=["x", "x", "y", "y", "z"],
                trait=["A", "C", "B", "C", "D"],
                location_side=[1, 1, -1, 1, 1],
            )
        ),
        fc.ResponseTable(
            pd.DataFrame(
                dict(
                    person_id=[1, 1, 1, 2, 2, 2],
                    item_id=["x", "y", "z"] * 2,
                    response=[0, 1, 0, 1, 0, 1],
                )
            ),
            format="item_long",
        ),
        model=spec,
        categories=(0, 1) if law.endswith("PL") else (0, 1, 2),
    )
    model = fc.Model(fc.PreparedInputs.from_inputs(source))
    constant = model.prepared.constant_data
    before = constant.copy(deep=True)
    values = {"theta": np.array([[0.1, -0.3, 0.2, 0.7], [0.5, 0.2, -0.1, -0.2]])}
    for layout in model.prepared.parameters:
        default = 1.2 if layout.kind == "loading" else 0.2
        if layout.kind == "upper_asymptote":
            default = 0.9
        value = np.where(np.isnan(layout.fixed), default, layout.fixed)
        if layout.kind == "location":
            value = np.where(
                np.isnan(layout.fixed),
                0.8 * layout.entry_data.orientation.values.reshape(layout.shape),
                layout.fixed,
            )
        values[layout.kind] = value
    point = labelled_point(model, values)
    xr.testing.assert_identical(model.prepared.data["constant_data"].to_dataset(), before)

    def evaluate(theta, *, indices, kernel, parameters):
        return kernel(theta[..., indices], *parameters)

    for batch, response in zip(model.prepared.batches, model._bindings.responses, strict=True):
        assert response.batch is batch
        # Resolve the item through its frozen unit, including unequal active counts.
        unit = response.batch.unit_index
        item = int(constant.unit_item.values[unit])
        active = np.flatnonzero(constant.item_q.values[item])
        assert batch.trait.shape[-1] == len(active)
        assert_allclose(response.batch.trait, np.broadcast_to(active, response.batch.trait.shape))
        aligned = model._bindings.evaluate_at(response.parameters, point)
        component = model.prepared.measurement_specs[unit].distribution
        full = {}
        record = constant.attrs["measurements"][unit]
        for slot in record["slots"]:
            entries = constant[slot["entry_index"]].values
            dense_values = np.zeros(entries.shape)
            present = entries >= 0
            dense_values[present] = point[slot["kind"]].values.ravel()[entries[present]]
            full[slot["key"]] = dense_values
        full = tuple(full[key] for key in component.parameter_keys)
        ability = jnp.asarray(point.theta.values[response.batch.person])

        sparse = partial(evaluate, indices=active, kernel=response.kernel, parameters=aligned)
        dense = partial(
            evaluate,
            indices=np.arange(4),
            parameters=full,
            kernel=component.kernel,
        )

        assert_allclose(sparse(ability), dense(ability), atol=1e-12)
        assert_allclose(jax.jacrev(sparse)(ability), jax.jacrev(dense)(ability), atol=1e-12)
        (density,) = model._bindings.evaluate_at(
            (pm.logp(response.distribution, response.batch.y),), point
        )
        expected = np.take_along_axis(
            np.asarray(dense(ability)), response.batch.y[:, None], axis=-1
        )
        assert_allclose(density, expected[:, 0], atol=1e-12)


def test_shared_latent_density_and_gradient_in_joint_response_time_model():
    with pm.Model() as model:
        latent = pm.Normal("latent", shape=2)
        response = TIRTDistribution.dist(
            pt.stack((latent[0], latent[1])),
            [1.2, 0.7],
            [0.0],
            statement_effect=[0.2, -0.1],
            residual_variance=[0.5, 0.5],
            link="probit",
        )
        model.register_rv(response, "response", observed=1)
        pm.LogNormal("rt", mu=2 - latent[0], sigma=0.3, observed=5.4)
    point = np.array([0.4, -0.2])
    contrast = 1.2 * point[0] - 0.7 * point[1] + 0.3
    expected = norm.logpdf(point).sum() + norm.logcdf(contrast)
    expected += norm.logpdf(np.log(5.4), 2 - point[0], 0.3) - np.log(5.4)
    assert_allclose(
        model.compile_logp(mode="FAST_COMPILE")({"latent": point}), expected, atol=1e-12
    )
    score = np.exp(norm.logpdf(contrast) - norm.logcdf(contrast))
    gradient = -point + score * np.array([1.2, -0.7])
    gradient[0] -= (np.log(5.4) - (2 - point[0])) / 0.3**2
    assert_allclose(
        model.compile_dlogp(mode="FAST_COMPILE")({"latent": point}), gradient, atol=1e-12
    )


def test_native_joint_prior_binds_batch_axes_and_preserves_correlated_event_gradient():
    from pyfcirt.models.priors import bind_prior, validate_prior

    mean = np.array([0.2, -0.3])
    covariance = np.array([[1.0, 0.4], [0.4, 0.7]])
    template = pm.MvNormal.dist(mu=mean, cov=covariance)
    validate_prior(template, role="coefficients", domain="real", shape=(2, 2))
    with pm.Model() as model:
        owned = bind_prior(template, name="coefficients", shape=(2, 2))
    point = np.array([[0.5, 0.1], [-0.2, 0.7]])
    delta = point - mean
    precision = np.linalg.inv(covariance)
    expected = -0.5 * np.einsum("ni,ij,nj->", delta, precision, delta)
    expected -= len(point) * (len(mean) * np.log(2 * np.pi) + np.linalg.slogdet(covariance)[1]) / 2
    assert owned.ndim == 2 and owned.owner.op.ndim_supp == 1
    assert_allclose(
        model.compile_logp(mode="FAST_COMPILE")({"coefficients": point}), expected, atol=1e-12
    )
    assert_allclose(
        model.compile_dlogp(mode="FAST_COMPILE")({"coefficients": point}),
        (-delta @ precision).ravel(),
        atol=1e-12,
    )


def test_ordinary_parameter_extension_reuses_preparation_graph_summary_and_scoring(monkeypatch):
    """A test-only statement shift changes only the declaration and response binding."""

    import pandas as pd

    import pyfcirt as fc
    from pyfcirt.models.mgppm import MGPPMDistribution
    from pyfcirt.models.specification import _normal_role, _Parameter
    from tests.data.test_schema import pair_input
    from tests.helpers import labelled_point, supplied_posterior

    monkeypatch.setattr(
        fc.MGPPM,
        "_parameters",
        (
            *fc.MGPPM._parameters,
            _Parameter(key="statement_shift", axes=("statement",), regression=True),
        ),
    )
    monkeypatch.setattr(
        fc.MGPPM, "defaults", {**fc.MGPPM.defaults, "statement_shift": _normal_role()}
    )
    original_bind = MGPPMDistribution.bind.__func__

    def bind(cls, spec, batch, *, ability, values):
        values = dict(values, intercept=values["intercept"] + values["statement_shift"])
        return original_bind(cls, spec, batch, ability=ability, values=values)

    monkeypatch.setattr(MGPPMDistribution, "bind", classmethod(bind))
    source = pair_input()
    model = fc.Model(fc.PreparedInputs.from_inputs(source))
    layout = next(p for p in model.prepared.parameters if p.kind == "statement_shift")
    assert layout.dims == ("statement_shift_id",) and layout.shape == (2,)
    np.testing.assert_array_equal(layout.free_flat, [0, 1])
    assert np.isnan(layout.fixed).all()
    model.add_regressions(
        "fc.statement_shift ~ 1", statement_data=pd.DataFrame({"statement_id": ["a", "b"]})
    )
    assert model.prepared.regressions[0].targets[0].kind == layout.kind
    model.set_priors({"fc.statement_shift": pm.Normal.dist(sigma=0.4)})
    assert not model.prepared.regressions
    point = labelled_point(
        model,
        {
            "population_mean": [[0.0, 0.0]],
            "covariance": np.eye(2),
            "loading": [1.0, 1.0],
            "statement_intercept": [0.0, 0.3],
            "statement_shift": [0.15, -0.2],
            "theta": [[0.4, -0.1]],
        },
    )
    response = model._bindings.responses[0]
    node = response.distribution.owner
    (actual,) = model._bindings.evaluate_at(node.op.dist_params(node), point)
    scores = response.batch.events.scores
    utility = np.array([0.4, -0.1]) - np.array([0.0, 0.3]) - np.array([0.15, -0.2])
    logits = scores @ utility
    expected = logits - np.logaddexp.reduce(logits)
    assert_allclose(actual, expected[None, response.batch.category_order], atol=1e-12)
    supplied_posterior(model, point, inference_method="mcmc", draws=range(12))
    snapshot = fc.parameter_point(model, include_person=False)
    assert_allclose(snapshot[layout.kind], point[layout.kind])
    table = fc.coef(model, var_names=["statement_shift"], kind="stats")
    assert len(table) == 2
    assert_allclose(table["mean"], [0.15, -0.2])
    scored = fc.score_model(model, source.responses)
    assert_allclose(scored.idata["fixed_parameters"][layout.kind], point[layout.kind])
    assert scored.build()[layout.kind] not in scored.build().free_RVs


def test_heterogeneous_gaussian_rank_shares_theta_and_has_distinct_evaluation_events():
    from dataclasses import replace

    import pandas as pd
    from pytensor.graph.traversal import ancestors

    import pyfcirt as fc
    from tests.helpers import mixed_inputs

    forced_choice, items = mixed_inputs()
    design = forced_choice.design
    rank_statements = ["rank_a", "rank_b", "rank_c"]
    forced_choice = replace(
        forced_choice,
        design=fc.FCDesignMapping(
            traits={**design.traits, "A": [*design.traits["A"], *rank_statements]},
            blocks={**design.blocks, "gaussian_rank": rank_statements},
            keying={**design.keying, "rank_b": -1},
        ),
        responses=fc.ResponseTable(
            pd.concat(
                [
                    forced_choice.responses.frame,
                    pd.DataFrame(
                        {
                            "person_id": [1, 2],
                            "block_id": ["gaussian_rank"] * 2,
                            "response": [[1, 3, 2], [2, 1, 3]],
                        }
                    ),
                ],
                ignore_index=True,
            ),
            format="block_long",
        ),
        blocks=(
            *forced_choice.blocks,
            fc.BlockSpec("gaussian_rank", model=fc.TIRTRank(loading_anchors={"rank_a": 1.0})),
        ),
    )
    model = fc.Model(fc.PreparedInputs.from_inputs((forced_choice, items)))
    calibration = model.build()
    bindings = model._bindings
    assert {type(spec).__name__ for spec in bindings.prepared.measurement_specs} >= {
        "MGPPM",
        "TIRTRank",
        "DichotomousIRT",
        "GPCM",
    }
    assert [rv.name for rv in calibration.free_RVs].count("theta") == 1
    assert all(
        calibration["theta"] in ancestors([response.distribution])
        for response in bindings.responses
    )
    rank_responses = [response for response in bindings.responses if response.augmentation]
    (rank_response,) = rank_responses
    contrasts, indicator = rank_response.augmentation
    assert not calibration.potentials
    assert indicator in calibration.observed_RVs
    assert contrasts in calibration.free_RVs
    assert contrasts in ancestors([indicator])
    assert np.isfinite(
        calibration.compile_logp(mode="JAX")(calibration.initial_point(random_seed=146))
    )

    evaluation = bindings.evaluation()
    assert evaluation.model is not calibration and not evaluation.model.potentials
    assert [rv.name for rv in evaluation.model.free_RVs].count("theta") == 1
    forward_rank = next(
        response for response in evaluation.responses if response.batch is rank_response.batch
    )
    assert not forward_rank.augmentation and not forward_rank.parameters
    assert indicator.name not in evaluation.model.named_vars
    assert contrasts.name not in evaluation.model.named_vars
    assert forward_rank.distribution.name == rank_response.batch.response_name
    assert contrasts.name not in {node.name for node in ancestors([forward_rank.distribution])}
    assert all(
        response.distribution in evaluation.model.observed_RVs for response in evaluation.responses
    )
    assert all(
        evaluation.model["theta"] in ancestors([response.distribution])
        and calibration["theta"] not in ancestors([response.distribution])
        for response in evaluation.responses
    )


def test_gaussian_rank_build_and_fit_keep_marginal_integration_out_of_calibration(monkeypatch):
    import xarray as xr
    from pytensor.graph.traversal import ancestors

    import pyfcirt as fc
    from pyfcirt.models.thurstonian import rank
    from tests.data.test_schema import pair_input
    from tests.helpers import labelled_point

    def forbidden(*args, **kwargs):
        raise AssertionError("Marginal integration entered calibration build or fit")

    calls = []

    def sample(**kwargs):
        assert pm.modelcontext(None) is calibration
        calls.append(kwargs)
        assert kwargs["idata_kwargs"]["log_likelihood"] is False
        assert all(not name.endswith("_raw") for name in kwargs["var_names"])
        posterior = labelled_point(model, {"theta": [[0.2, -0.1]]}).expand_dims(chain=[0], draw=[0])
        return xr.DataTree.from_dict({"posterior": posterior})

    inputs_seen = []
    log_probabilities = rank._GaussianEvent.log_probabilities

    def record_inputs(event, *inputs):
        inputs_seen.append(inputs)
        return log_probabilities(event, *inputs)

    monkeypatch.setattr(rank._GaussianEvent, "log_probabilities", record_inputs)
    with monkeypatch.context() as guard:
        guard.setattr(rank, "sobol_nodes", forbidden)
        guard.setattr(rank.TIRTRankDistribution, "_log_probs", forbidden)
        guard.setattr(pm, "sample", sample)
        source = pair_input(model=fc.TIRTRank(loading_anchors={"a": 1.0, "b": 0.8}))
        model = fc.Model(fc.PreparedInputs.from_inputs(source))
        calibration = model.build()
        nodes = dict(calibration.named_vars)
        observed_rvs = tuple(calibration.observed_RVs)
        free_rvs = tuple(calibration.free_RVs)
        point = calibration.initial_point(random_seed=146)
        target = calibration.compile_logp(mode="JAX")
        density = target(point)
        assert np.isfinite(density)
        fitted = model.fit(
            draws=1, tune=0, chains=1, nuts_sampler=None, idata_kwargs={"log_likelihood": True}
        )
        assert "log_likelihood" not in fitted
        assert len(calls) == 1 and model.build() is calibration
        assert model._bindings._evaluation is None

    response = model._bindings.responses[0]
    batch = response.batch
    requested = [f"{batch.name}_log_probability", f"{batch.name}_log_likelihood"]
    evaluation = model._bindings.evaluation(var_names=[f"{batch.name}_p"])
    assert model._bindings.evaluation(var_names=requested[:1]) is evaluation
    (actual_inputs,) = inputs_seen
    response_rv = evaluation.responses[0].distribution
    native_node = response_rv.owner
    expected_inputs = native_node.op.dist_params(native_node)
    assert all(
        actual is expected for actual, expected in zip(actual_inputs, expected_inputs, strict=True)
    )
    assert model._bindings.evaluation(var_names=requested[1:]) is evaluation
    assert len(inputs_seen) == 1
    assert evaluation.model[requested[0]] in set(ancestors([evaluation.model[requested[1]]]))
    assert evaluation.model is not calibration and not evaluation.model.potentials
    assert response_rv in evaluation.model.observed_RVs
    for name in (*requested, f"{batch.name}_p"):
        assert name in evaluation.model.named_vars and name not in calibration.named_vars
        inputs = set(ancestors([evaluation.model[name]]))
        assert evaluation.model["theta"] in inputs
        assert not set(free_rvs).intersection(inputs)
        assert not set(evaluation.responses[0].augmentation).intersection(inputs)
    assert dict(calibration.named_vars) == nodes
    assert tuple(calibration.observed_RVs) == observed_rvs
    assert not calibration.potentials
    assert tuple(calibration.free_RVs) == free_rvs
    assert_allclose(calibration.compile_logp(mode="JAX")(point), density, atol=1e-12)
