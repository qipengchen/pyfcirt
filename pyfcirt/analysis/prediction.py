"""Evaluate the existing response graph with native PyMC and labelled final samples."""

from __future__ import annotations

from numbers import Integral
from secrets import randbits
from typing import TYPE_CHECKING
from uuid import uuid4

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr

from pyfcirt.analysis.output import pool_log_likelihood, posterior_samples, prepare_output
from pyfcirt.common.validation import has_labelled_axes

if TYPE_CHECKING:
    from pyfcirt.model import Model

_SOURCE_COORDS = ("source_chain", "source_draw")


def _seed(value):
    if value is not None and (
        isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0
    ):
        raise ValueError("random_seed must be None or a nonnegative integer")
    return None if value is None else int(value)


def _jax_key(seed):
    """Fold every word of a validated seed; no arbitrary integer width limit."""
    key = jax.random.key(0)
    for shift in range(0, max(1, seed.bit_length()), 32):
        key = jax.random.fold_in(key, (seed >> shift) & 0xFFFFFFFF)
    return key


def _select_states(posterior: xr.Dataset, draws: int | None, seed) -> xr.Dataset:
    if draws is None:
        return posterior
    if isinstance(draws, (bool, np.bool_)) or not isinstance(draws, Integral) or draws <= 0:
        raise ValueError("draws must be a positive integer or None")
    chains, per_chain = posterior.sizes["chain"], posterior.sizes["draw"]
    if draws > chains * per_chain:
        raise ValueError("draws exceeds the available posterior states")
    keys = jax.random.split(_jax_key(randbits(32) if seed is None else seed), chains + 1)
    counts = jnp.full(chains, draws // chains, dtype=int)
    counts = counts.at[jax.random.permutation(keys[0], chains)[: draws % chains]].add(1)
    counts = np.asarray(counts)  # Host indices select xarray labels, not numerical model values.
    chain = np.repeat(np.arange(chains), counts)
    draw = np.asarray(
        jnp.concatenate(
            [
                jax.random.permutation(key, per_chain)[:count]
                for key, count in zip(keys[1:], counts, strict=True)
            ]
        )
    )
    return _take_states(posterior, chain * per_chain + draw, chains=[0], draws=np.arange(draws))


def _take_states(posterior, positions, *, chains, draws):
    """Select the stacked sample axis and keep its original labels."""
    selected = posterior.stack(state=("chain", "draw"), create_index=False).isel(state=positions)
    return (
        selected.assign_coords(
            state=draws,
            source_chain=selected.variables["chain"],
            source_draw=selected.variables["draw"],
        )
        .drop_vars(["chain", "draw"])
        .rename(state="draw")
        .expand_dims(chain=chains)
    )


def _native_forward(
    bindings, posterior, names, *, sample_names=(), draws=None, random_seed=None, progressbar=True
):
    import pymc as pm

    conditioned = []
    if names:
        bindings = bindings.evaluation(var_names=names)
        conditioned = bindings.conditioned_names(
            [bindings.model[name] for name in names], posterior
        )
    final = posterior[conditioned].assign_coords(
        {
            axis: posterior[axis]
            for axis in ("chain", "draw", *_SOURCE_COORDS)
            if axis in posterior.coords
        }
    )
    final = _select_states(final, draws, random_seed)
    if not names:
        return final
    # PyMC's trace converter reads sample axes from its first array.
    trace = (
        final
        if final.data_vars
        else final.assign(__draw_label=final.draw.expand_dims(chain=final.chain))
    )
    sampled = pm.sample_posterior_predictive(
        trace,
        model=bindings.model,
        var_names=list(names),
        sample_vars=list(sample_names),
        freeze_vars=list(final.data_vars),
        random_seed=random_seed,
        progressbar=progressbar,
        extend_inferencedata=False,
        idata_kwargs=bindings.idata_kwargs(),
    )["posterior_predictive"].to_dataset()
    for name in _SOURCE_COORDS:
        if name in final.coords:
            sampled = sampled.assign_coords({name: final[name]})
    return sampled


def _prediction_result(model, groups, *, source, attrs, store, save_responses=True):
    """Publish prepared observations and support with prior or posterior predictions."""
    prepared = model.prepared
    predictive = f"{source}_predictive"
    for dataset in groups.values():
        dataset.attrs = dict(attrs)
    observed = prepared.observed_data
    names = [name for name in groups[predictive] if name in observed]
    if names:
        groups["observed_data"] = observed[names]
    stored = set()
    if store:
        selected = groups if save_responses else {"prior": groups["prior"]}
        remove = (
            ("prior", predictive)
            if source == "prior"
            else (
                predictive,
                "response_probabilities",
                "response_parameters",
                "posterior_predictive_checks",
                "discrepancy",
            )
        )
        model._publish_outputs(selected, remove=remove)
        stored = selected.keys()
    groups = {
        name: model.idata[name].to_dataset()
        if name in stored
        else prepare_output(prepared, ds, name)
        for name, ds in groups.items()
    }
    design = prepared.response_metadata["design"]
    groups["design"] = design[["unit_block", "unit_item", "obs_unit", "obs_batch"]].assign_coords(
        block=design.block, item=design.item
    )
    groups.update(
        (path, support)
        for path, support in prepared.response_metadata.items()
        if path.startswith("response_support/")
    )
    return xr.DataTree.from_dict(groups, name=f"{source}_prediction")


def _sample_responses(model, posterior, batches, outputs, *, draws, seed, progressbar):
    """Draw selected response RVs and graph outputs at validated posterior states."""
    response_names = [batch.response_name for batch in batches]
    evaluated = _native_forward(
        model._bindings,
        posterior,
        [*response_names, *outputs],
        sample_names=response_names,
        draws=draws,
        random_seed=seed,
        progressbar=progressbar,
    )
    attrs = dict(
        prediction_run_id=uuid4().hex,
        posterior_selection="all" if draws is None else "balanced random without replacement",
    )
    if "posterior_run_id" in model.idata.attrs:
        attrs["posterior_run_id"] = model.idata.attrs["posterior_run_id"]
    evaluated.attrs = attrs
    return evaluated


def posterior_predictive(
    model: Model,
    *,
    draws=None,
    random_seed=None,
    include_probabilities=True,
    store=True,
    progressbar=True,
) -> xr.DataTree:
    """Draw responses and requested probability/shape outputs in one native forward operation."""
    if not isinstance(include_probabilities, bool) or not isinstance(store, bool):
        raise TypeError("include_probabilities and store must be boolean")
    seed = _seed(random_seed)
    posterior = posterior_samples(model.idata, mcmc=True)
    batches = model.prepared.batches
    optional = {}
    if include_probabilities:
        for batch in batches:
            group = (
                "response_parameters"
                if batch.likelihood_kind == "beta"
                else "response_probabilities"
            )
            optional.setdefault(group, []).extend(batch.distribution_names)
    evaluated = _sample_responses(
        model,
        posterior,
        batches,
        [name for names in optional.values() for name in names],
        draws=draws,
        seed=seed,
        progressbar=progressbar,
    )
    groups = {"posterior_predictive": evaluated[[batch.response_name for batch in batches]]}
    groups.update({group: evaluated[names] for group, names in optional.items()})
    return _prediction_result(model, groups, source="posterior", attrs=evaluated.attrs, store=store)


def prior_predictive(
    model: Model, *, draws=500, random_seed=None, save_responses=False, store=True
) -> xr.DataTree:
    """Sample actual parameter and response priors, excluding calibration auxiliaries."""
    if isinstance(draws, (bool, np.bool_)) or not isinstance(draws, Integral) or draws <= 0:
        raise ValueError("draws must be a positive integer")
    if not isinstance(save_responses, bool) or not isinstance(store, bool):
        raise TypeError("save_responses and store must be boolean")
    conditioning = model._prior_conditioning()
    bindings = model._bindings.evaluation()
    response_names = [batch.response_name for batch in model.prepared.batches]
    parameter_names = list(bindings.parameters)
    sampled = bindings.prior_draws(draws=int(draws), random_seed=_seed(random_seed))
    groups = dict(prior=sampled[parameter_names], prior_predictive=sampled[response_names])
    return _prediction_result(
        model,
        groups,
        source="prior",
        attrs={"conditioning": conditioning},
        store=store,
        save_responses=save_responses,
    )


def _response_log_densities(bindings, posterior, batches, *, progressbar):
    """Evaluate existing marginal expressions with the same final parameter values."""
    nodes = {batch.log_likelihood_name: batch.response_name for batch in batches}
    return _native_forward(bindings, posterior, nodes, progressbar=progressbar).rename(nodes)


def log_likelihood(model: Model, *, store=True, progressbar=True) -> xr.DataArray:
    """Evaluate missing marginal factors at the current posterior's final nodes."""
    posterior = posterior_samples(model.idata, mcmc=True)
    prepared = model.prepared
    obs = prepared.constant_data.get_index("obs")
    cached = positions = None
    if (group := model.idata.get("log_likelihood")) is not None:
        stored = group["response"]
        if has_labelled_axes(stored, ("chain", "draw", "obs")) and all(
            stored.indexes[axis].equals(posterior.indexes[axis]) for axis in ("chain", "draw")
        ):
            labels = stored.indexes["obs"]
            if stored.attrs.get("complete_observations", False) and labels.equals(obs):
                return stored
            known = obs.get_indexer(labels)
            if (known >= 0).all():
                positions = np.full(len(obs), -1, dtype=np.intp)
                positions[known] = np.arange(len(known), dtype=np.intp)
                cached = stored.reset_coords(drop=True)
    batches = prepared.batches
    if not batches or any(batch.y is None for batch in batches):
        raise ValueError("Log likelihood requires actual observed responses")
    missing = [
        batch for batch in batches if positions is None or (positions[batch.obs_slice] < 0).any()
    ]
    values = (
        _response_log_densities(model._bindings, posterior, missing, progressbar=progressbar)
        if missing
        else {}
    )
    result = pool_log_likelihood(prepared, values, cached=cached, positions=positions).to_dataset()
    if store:
        model._publish_outputs({"log_likelihood": result})
        return model.idata["log_likelihood"]["response"]
    return prepare_output(model.prepared, result, "log_likelihood")["response"]


def predictive_view(idata: xr.DataTree, *, block=None, item=None) -> xr.DataTree:
    """Select real observed/replicated responses for one compatible block or item."""
    from pyfcirt.data.coordinates import select_units

    if not isinstance(idata, xr.DataTree):
        raise TypeError("idata must be a predictive DataTree")
    if (block is None) == (item is None):
        raise ValueError("Select exactly one block or item")
    axis, unit = ("block", block) if block is not None else ("item", item)
    group = "posterior_predictive" if "posterior_predictive" in idata else "prior_predictive"
    if group not in idata:
        raise ValueError("idata has no predictive responses")
    if not {"design", "response_support"} <= idata.children.keys():
        raise ValueError("Regenerate predictions to attach their prepared response support")
    design = idata["design"].to_dataset()
    values = idata[group].to_dataset()
    units = (
        select_units(design, blocks=[unit])
        if axis == "block"
        else select_units(design, items=[unit])
    )
    if not {"obs_unit", "obs_batch"} <= design.data_vars.keys():
        raise ValueError("Regenerate predictions to attach their prepared observation catalogue")
    observations = design.obs_batch.isel(obs=np.flatnonzero(np.isin(design.obs_unit, units)))
    batch_ids = np.unique(observations.values)
    if not len(batch_ids):
        raise ValueError(f"Unknown {axis} {unit!r} in predictions")
    if len(batch_ids) != 1:
        # Preparation creates one batch for each distinct response support of a unit.
        raise ValueError("Predictive plots require a consistent support within the selected unit")
    support = next(
        (
            node
            for node in idata["response_support"].children.values()
            if node.attrs.get("batch_index") == batch_ids[0]
        ),
        None,
    )
    if support is None or support.attrs.get("response_name") not in values:
        raise ValueError("Regenerate predictions to attach their prepared response support")
    name, dim = (support.attrs[field] for field in ("response_name", "obs_dim"))
    if "observed_data" not in idata or name not in idata["observed_data"]:
        raise ValueError("This prediction has no actual observed responses")

    def view(dataset):
        response = dataset[name].rename({dim: "obs"}).sel(obs=observations.obs)
        return response.reset_coords(drop=True).rename("response").to_dataset()

    kind = support.attrs["likelihood_kind"]
    attrs = {"response_kind": kind}
    if kind != "beta":
        count = support.sizes[support.attrs["category_dim"]]
        attrs["data_type"] = (
            "binary"
            if count == 2
            else "categorical"
            if support.attrs["response_format"]
            else "ordinal"
        )
    return xr.DataTree.from_dict(
        {
            "/": xr.Dataset(attrs=attrs),
            "observed_data": view(idata["observed_data"].to_dataset()),
            group: view(values),
        },
        name="predictive_view",
    )
