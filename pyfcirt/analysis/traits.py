"""Marginal population moments from saved final nodes and frozen current-person designs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import xarray as xr

from pyfcirt.analysis.summary import _fixed_entries, _summary_options, _summary_samples

if TYPE_CHECKING:
    from pyfcirt.model import Model


def _fixed_moments(constant, nodes, selected, regression):
    """Propagate known constraints, including cancellation of common person means."""
    groups = constant.person_group.values[selected]
    present = np.unique(groups)
    means, mean_known = nodes["population_mean"]
    covariance, covariance_known = nodes["covariance"]
    trait_count = constant.sizes["trait"]
    person_mean = means[groups].copy()
    mean_fixed = mean_known[present].all(axis=0)
    centered_fixed = mean_fixed.copy() if len(present) > 1 else np.ones(trait_count, dtype=bool)
    if regression is not None:
        matrix, rows, pools, positions = regression
        matrix, rows, pools = matrix, rows[selected], pools[selected]
        coefficients, coefficient_known = nodes["regression_coefficient"]
        coefficients = coefficients.ravel()[positions]
        coefficient_known = coefficient_known.ravel()[positions]
        for trait in range(trait_count):
            x = matrix[rows[:, trait]]
            indices = pools[:, trait]
            person_mean[:, trait] += (x * coefficients[indices]).sum(axis=-1)
            for pool in np.unique(indices):
                weights = x * (indices == pool)[:, None]
                unknown = ~coefficient_known[pool]
                mean_fixed[trait] &= not np.any(weights.mean(axis=0)[unknown] != 0)
                centered_fixed[trait] &= not np.any(
                    (weights - weights.mean(axis=0))[:, unknown] != 0
                )
    centered = person_mean - person_mean.mean(axis=0)
    zero = centered_fixed & np.all(centered == 0, axis=0)
    residual_fixed = (
        covariance_known[present].all(axis=0) if covariance.ndim == 3 else covariance_known.copy()
    )
    spread_fixed = (
        (centered_fixed[:, None] & centered_fixed[None, :]) | zero[:, None] | zero[None, :]
    )
    total_fixed = residual_fixed & spread_fixed
    sd_fixed = np.diag(total_fixed).copy()
    correlation_fixed = total_fixed & sd_fixed[:, None] & sd_fixed[None, :]
    np.fill_diagonal(correlation_fixed, True)
    return mean_fixed, sd_fixed, correlation_fixed


def trait_summary(
    data: Model,
    *,
    idata: xr.DataTree | None = None,
    by_group: bool = False,
    ci_prob: float = 0.95,
    ci_kind: Literal["hdi", "eti"] = "hdi",
    **summary_kwargs,
) -> xr.Dataset:
    """Summarize per-draw marginal mean, SD and correlation for current people.

    The reference distribution is the current person's frozen group and covariate
    distribution. Posterior transformations precede native ArviZ statistics.
    Sample coordinates select original posterior labels before transformation.
    """
    import jax
    import jax.numpy as jnp
    from arviz_base import extract
    from arviz_stats import summary

    from pyfcirt.covariates.model import indexed_mean
    from pyfcirt.models.population import PopulationFactory

    if type(by_group) is not bool:
        raise TypeError("by_group must be a bool")
    from pyfcirt.model import Model

    if not isinstance(data, Model):
        raise TypeError("trait_summary requires the current Model for population layouts")
    posterior, method = _summary_samples(data, idata)
    prepared = data.prepared
    constant = prepared.constant_data
    if not constant.sizes["person"]:
        raise ValueError("Marginal moments require at least one current person")
    options, statistics = _summary_options(
        summary_kwargs, fmt="xarray", ci_prob=ci_prob, ci_kind=ci_kind
    )
    selection = {key: options.pop(key, None) for key in ("var_names", "filter_vars", "coords")}
    recipe = prepared.population.recipe
    grouped = by_group and bool(recipe["groups"])
    indices = (
        [
            np.flatnonzero(constant.person_group.values == index)
            for index in range(constant.sizes["population_group"])
        ]
        if grouped
        else [np.arange(constant.sizes["person"])]
    )
    for index, selected in enumerate(indices):
        if not len(selected):
            raise ValueError(
                f"Population group {constant.population_group.values[index]!r} has no people"
            )
    layouts = prepared.parameter_layouts
    required = ["population_mean", "covariance"]
    regression = None
    coefficient_entries = None
    if (plan := prepared.theta_regression) is not None:
        (binding,) = plan.targets
        # Other regressions can share the kind without affecting population moments.
        positions = plan.coefficient_entries
        coefficient_entries, local_positions = np.unique(positions, return_inverse=True)
        regression = (
            plan.design.X.values,
            binding.row_index,
            binding.pool_index,
            local_positions.reshape(positions.shape),
        )
        required.append("regression_coefficient")
    nodes = {key: _fixed_entries(layouts[key], prepared.fixed_parameters) for key in required}
    if coefficient_entries is not None:
        nodes["regression_coefficient"] = tuple(
            value.ravel()[coefficient_entries] for value in nodes["regression_coefficient"]
        )
    # Unknown reference values are zero only for evaluating invariant outputs.
    nodes = {key: (np.where(known, value, 0), known) for key, (value, known) in nodes.items()}
    coords = selection["coords"] or {}
    sample_selection = {dim: labels for dim, labels in coords.items() if dim in {"chain", "draw"}}
    if sample_selection:
        if not {"chain", "draw"} <= posterior.coords.keys():
            raise ValueError("Sample selection requires a labelled posterior")
        # Preserve source labels and transform only the requested joint states.
        posterior = posterior.sel(sample_selection)
        # Native scalar selection keeps the label as a coordinate; restore its sample axis.
        posterior = posterior.expand_dims(
            [dim for dim in ("chain", "draw") if dim not in posterior.dims]
        )
        if any(posterior.sizes[dim] == 0 for dim in ("chain", "draw")):
            raise ValueError("Sample selection must retain at least one chain and draw")
    trait_count = constant.sizes["trait"]
    shapes = ((trait_count,), (trait_count,), (trait_count, trait_count))
    names = ("trait_mean", "trait_sd", "trait_correlation")
    dims = (("trait",), ("trait",), ("trait", "trait_aux"))
    coordinates = {"trait": constant.trait.values, "trait_aux": constant.trait_aux.values}
    if grouped:
        coordinates["population_group"] = constant.population_group.values
    axes = [(("population_group",) if grouped else ()) + dimensions for dimensions in dims]
    shapes = [((len(indices),) if grouped else ()) + shape for shape in shapes]
    # Select integer output positions first, without chain/draw-sized placeholders.
    selectors = extract(
        xr.Dataset(
            {
                name: (dimensions, np.arange(int(np.prod(shape))).reshape(shape))
                for name, shape, dimensions in zip(names, shapes, axes, strict=True)
            },
            coords=coordinates,
        ),
        var_names=selection["var_names"],
        filter_vars=selection["filter_vars"],
        combined=False,
        keep_dataset=True,
        sample_dims=[],
    ).sel({dim: labels for dim, labels in coords.items() if dim not in {"chain", "draw"}})
    output_positions = {name: value.values.ravel() for name, value in selectors.data_vars.items()}
    moment_index = {name: index for index, name in enumerate(names)}
    known = [_fixed_moments(constant, nodes, selected, regression) for selected in indices]
    fixed_entries = {
        name: np.stack([flags[moment_index[name]] for flags in known]).ravel()[positions]
        for name, positions in output_positions.items()
    }
    fixed_positions, free_positions = {}, {}
    for name, mask in fixed_entries.items():
        positions = output_positions[name]
        if mask.any():
            fixed_positions[name] = positions[mask]
        if not mask.all():
            free_positions[name] = positions[~mask]
    stochastic = [
        key
        for key in required
        if free_positions
        and not nodes[key][1].all()
        and (key != "covariance" or free_positions.keys() != {"trait_mean"})
    ]
    if stochastic and method != "mcmc":
        raise NotImplementedError("Trait summaries require MCMC posterior samples")
    for key in stochastic:
        if key not in posterior:
            raise ValueError(f"Missing required final posterior node: {key}")
    group_index = jnp.asarray(constant.person_group.values)

    def moments(positions, *values):
        final = dict(zip(required, values, strict=True))
        regression_mean = None
        if regression is not None:
            matrix, rows, pools, entries = regression
            coefficients = final["regression_coefficient"].ravel()[entries]
            regression_mean = indexed_mean(matrix, coefficients, rows, pools)
        person_mean, person_covariance = PopulationFactory.moments(
            group_index,
            final["population_mean"],
            final["covariance"],
            regression_mean=regression_mean,
        )
        group_moments = [
            PopulationFactory.marginal_moments(person_mean[selected], person_covariance[selected])
            for selected in indices
        ]
        return tuple(
            jnp.stack([entry[moment_index[name]] for entry in group_moments]).ravel()[index]
            for name, index in positions.items()
        )

    fixed_values = {}
    if fixed_positions:
        with jax.enable_x64():
            known = jax.jit(lambda *values: moments(fixed_positions, *values))(
                *(nodes[key][0] for key in required)
            )
        fixed_values = dict(zip(fixed_positions, known, strict=True))
    chain_count = posterior.sizes["chain"] if free_positions else 0
    draw_count = posterior.sizes["draw"] if free_positions else 0
    output = {
        name: np.empty((chain_count, draw_count, len(positions)))
        for name, positions in free_positions.items()
    }
    transform = jax.jit(jax.vmap(lambda *values: moments(free_positions, *values)))
    # Keep person-by-trait/matrix intermediates bounded; retain only selected free draws.
    predictor_count = 0 if regression is None else regression[0].shape[1]
    cells_per_draw = constant.sizes["person"] * trait_count * (trait_count + predictor_count)
    batch_size = max(1, min(32, 262144 // cells_per_draw))
    coefficient_index = (
        {layouts["regression_coefficient"].dims[0]: coefficient_entries}
        if coefficient_entries is not None
        else {}
    )
    for chain in range(chain_count):
        for start in range(0, draw_count, batch_size):
            stop = min(start + batch_size, draw_count)
            inputs = []
            for key in required:
                value, mask = nodes[key]
                if key in stochastic:
                    indexers = coefficient_index if key == "regression_coefficient" else {}
                    draws = (
                        posterior[key]
                        .isel(chain=chain, draw=slice(start, stop), **indexers)
                        .transpose("draw", *layouts[key].dims)
                        .values
                    )
                    inputs.append(np.where(mask, value, draws))
                else:
                    inputs.append(np.broadcast_to(value, (stop - start, *value.shape)))
            with jax.enable_x64():
                transformed = transform(*inputs)
            for name, values in zip(output, transformed, strict=True):
                output[name][chain, start:stop] = np.asarray(values)
    columns = statistics
    if "mean" not in columns and fixed_values:
        columns = ["mean", *columns]
    if output:
        native = summary(
            xr.Dataset(
                {
                    name: (("chain", "draw", f"{name}_entry"), draws)
                    for name, draws in output.items()
                },
                coords={dim: posterior[dim] for dim in ("chain", "draw")},
            ),
            fmt="xarray",
            sample_dims=["chain", "draw"],
            **options,
        ).reindex(summary=columns)
    result = {}
    for name, selector in selectors.data_vars.items():
        mask = fixed_entries[name]
        table = np.full((selector.size, len(columns)), np.nan)
        if name in output:
            table[~mask] = native[name].transpose(f"{name}_entry", "summary").values
        if name in fixed_values:
            table[mask, columns.index("mean")] = fixed_values[name]
        result[name] = xr.DataArray(
            table.reshape((*selector.shape, len(columns))),
            dims=(*selector.dims, "summary"),
            coords={**{dim: selector[dim] for dim in selector.dims}, "summary": columns},
        )
    return xr.Dataset(
        result, attrs=dict(target="marginal_population", reference_distribution="current_persons")
    )
