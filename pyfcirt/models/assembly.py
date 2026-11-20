"""One runtime owner for native model construction and cached evaluation views."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from threading import Lock

import jax
import jax.numpy as jnp
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from pymc.model.fgraph import fgraph_from_model, model_from_fgraph
from pymc.pytensorf import rvs_in_graph
from pytensor import config, wrap_jax
from pytensor.compile.sharedvalue import SharedVariable
from pytensor.graph.replace import graph_replace
from pytensor.graph.traversal import ancestors, graph_inputs

from pyfcirt.covariates.model import build_regressions
from pyfcirt.data.coordinates import _identity_digest
from pyfcirt.models.categorical import StableCategorical
from pyfcirt.models.graph import ParameterFactory, ResponseBinding, register_response
from pyfcirt.models.population import MARGINAL_CORRELATION, PopulationFactory
from pyfcirt.models.priors import prior_coordinates


@dataclass(slots=True, kw_only=True)
class ModelGraph:
    """Hold actual nodes once; graph objects and caches never become saved state."""

    model: pm.Model
    prepared: object
    parameters: dict
    responses: tuple[ResponseBinding, ...] = ()
    _calibration_graph: tuple[pm.Model, dict] | None = field(default=None, repr=False)
    _evaluation: ModelGraph | None = field(default=None, repr=False)
    _stochastic_finals: dict | None = field(default=None, init=False, repr=False)
    _point_program: tuple | None = field(default=None, init=False, repr=False)
    _point_lock: Lock = field(default_factory=Lock, init=False, repr=False)

    @classmethod
    def build(cls, prepared, *, approximation=None):
        """Assemble frozen scopes; this is the only actual PyMC construction path."""
        recipe = prepared.population.recipe
        if recipe["regression_by_group"] and recipe["regression"] is None:
            raise ValueError("regression_by_group requires a theta Regression")
        with pm.Model(coords=prepared.coords) as model:
            factory = ParameterFactory(prepared)
            factory.build(
                kind
                for kind in ("regression_coefficient", "regression_sd")
                if kind in factory.layouts
            )
            regressions = build_regressions(prepared.regressions, factory.bindings)
            factory.build(
                (kind for kind in factory.layouts if kind != "covariance"), regressions=regressions
            )
            theta_regression = None
            if (plan := prepared.theta_regression) is not None:
                (target,) = plan.targets
                theta_regression = regressions[target.binding_id, target.prior_role]
            person_mean, person_covariance = PopulationFactory.build(
                prepared.population,
                parameter_factory=factory,
                regression=theta_regression,
            )
            theta = factory.bindings["theta"]
            if prepared.constant_data.attrs["marginal_correlation"]:
                pm.Deterministic(
                    MARGINAL_CORRELATION,
                    wrap_jax(PopulationFactory.marginal_moments, allow_eval=False)(
                        person_mean, person_covariance
                    )[2],
                    dims=("trait", "trait_aux"),
                )
            graph = cls(
                model=model,
                prepared=prepared,
                parameters=factory.bindings,
            )
            graph.responses = tuple(
                batch.specification.distribution.bind(
                    batch.specification,
                    batch,
                    ability=theta[batch.person[:, None], batch.trait],
                    parameters=graph.parameters,
                )
                for batch in prepared.batches
            )
        if prepared.fixed_parameters is None and approximation is None:
            return graph
        roots = {theta.name}
        if MARGINAL_CORRELATION in model.named_vars:
            roots.add(MARGINAL_CORRELATION)
        fixed = {
            name: pt.constant(value, dtype=model[name].dtype)
            for name, value in prepared.intervention_values.items()
        }
        roots.update(fixed)
        # Final point values must remain available even for units with no answers.
        roots.update(kind for kind, layout in factory.layouts.items() if layout.point_parameter)
        if approximation is None:
            conditioned = pm.do(model, fixed, make_interventions_shared=False)
        else:
            conditioned = approximation._apply(model, fixed)
            roots.update(approximation.var_names)
        roots.update(node.name for node in (*conditioned.observed_RVs, *conditioned.potentials))
        updated = graph._rebind(_retain_outputs(conditioned, roots))
        if approximation is not None:
            updated._calibration_graph = model, fixed
        return updated

    @staticmethod
    def calibration_coordinates(prepared):
        """Identify shared native bindings independently of the current respondents."""
        shared = {layout.kind for layout in prepared.shared_parameters}
        coordinates = [
            (
                block.binding_id,
                block.noncentered,
                block.scale_kind,
                block.scale_power,
                role,
                positions,
                block.prior_shapes[role],
                prior_coordinates(
                    block.priors[role],
                    noncentered=block.noncentered
                    and block.scale_kind is None
                    and role not in block.centered_roles,
                ),
            )
            for block in prepared.blocks
            if block.kind in shared and len(block.free_flat)
            for role, positions in block.prior_groups
            if role in block.priors
        ]
        return _identity_digest(("native-calibration-v1", config.floatX, coordinates))

    def calibration_source(self):
        """Return the source model and its shared final-to-free dependency relation."""
        native = self.model
        if self._calibration_graph is not None:
            native, fixed = self._calibration_graph
            if fixed:
                native = pm.do(native, fixed, make_interventions_shared=False)
                self._calibration_graph = native, {}
        free = set(native.free_RVs)
        return native, {
            node: free.intersection(ancestors([node]))
            for layout in self.prepared.shared_parameters
            if (node := native.named_vars.get(layout.kind)) is not None
        }

    @staticmethod
    def calibration_groups(native, dependencies):
        """Assign each supported RV to its nearest shared final parameter kind."""
        groups, assigned = {}, set()
        finals = set(dependencies)
        for node, upstream in dependencies.items():
            # Stop at other final parameters, including a dependent bound's parent.
            direct = upstream.intersection(ancestors([node], blockers=finals - {node}))
            members = [
                rv
                for rv in native.free_RVs
                if rv in direct - assigned and np.dtype(rv.dtype).kind == "f"
            ]
            if members:
                groups[node.name] = [rv.name for rv in members]
                assigned.update(members)
        return groups

    @staticmethod
    def calibration_complement(native, dependencies, selected):
        """Find final nodes or uncovered native RVs to freeze around a prior replacement."""
        updated = {native[name] for name in selected}
        frozen = {node for node, rvs in dependencies.items() if rvs and rvs.isdisjoint(updated)}
        blocks = set(frozen)
        covariance = native["covariance"]
        components = {native["trait_correlation"], native["population_sd"]}
        if covariance in frozen:
            # A fixed covariance also supplies these two reporting components.
            frozen -= components
            blocks |= components
        remaining = set(native.free_RVs).intersection(ancestors(dependencies, blockers=blocks))
        remaining -= updated | blocks
        return {node.name for node in frozen}, {node.name for node in remaining}

    def idata_kwargs(self, options=None):
        """Keep frozen label types at native PyMC conversion boundaries."""
        options = {} if options is None else dict(options)
        options["coords"] = {**(options.get("coords") or {}), **self.prepared.coords}
        return options

    def prior_draws(self, *, draws, random_seed):
        """Draw final variables and responses together with their native coordinates."""
        responses = [response.batch.response_name for response in self.responses]
        sampled = pm.sample_prior_predictive(
            draws=draws,
            model=self.model,
            var_names=[*self.parameters, *responses],
            random_seed=random_seed,
            idata_kwargs=self.idata_kwargs(),
        )
        return xr.merge(
            (
                sampled[name].to_dataset()
                for name in ("prior", "prior_predictive")
                if name in sampled
            ),
            compat="no_conflicts",
        )

    def conditioned_names(self, outputs, values):
        """Select sampled final nodes on the path to these outputs."""
        if self._stochastic_finals is None:
            free = set(self.model.free_RVs)
            self._stochastic_finals = {
                name: node
                for name, node in self.parameters.items()
                if not free.isdisjoint(ancestors([node]))
            }
        needed = set(ancestors(outputs, blockers=self.parameters.values()))
        names = [name for name, node in self._stochastic_finals.items() if node in needed]
        if missing := set(names) - values.data_vars.keys():
            raise ValueError(f"Missing required final posterior nodes: {sorted(missing)}")
        return names

    def evaluate_at(self, outputs, point):
        """Evaluate native expressions at a labelled point, without drawing raw nodes."""
        if not isinstance(point, xr.Dataset) or {"chain", "draw"}.intersection(point.dims):
            raise ValueError("point must be a final-parameter Dataset without sample dimensions")
        outputs = tuple(outputs)
        names = self.conditioned_names(outputs, point)
        inputs, values = [], []
        for name in names:
            node = self.parameters[name]
            value = np.asarray(point[name].values)
            tensor = pt.tensor(dtype=node.dtype, shape=value.shape, name=f"{name}_value")
            # Points saved on another device may differ from graph precision.
            with np.errstate(over="ignore", invalid="ignore"):
                value = tensor.type.filter(value, allow_downcast=True)
            if not np.isfinite(value).all():
                raise ValueError(
                    f"{name}: values cannot be represented as graph dtype {node.dtype}"
                )
            inputs.append(tensor)
            values.append(value)
        signature = (
            outputs,
            tuple(zip(names, (tensor.type for tensor in inputs), strict=True)),
            config.floatX,
            jax.config.x64_enabled,
            jax.config.jax_default_device,
        )
        # PyTensor's callable owns mutable input storage. Reuse the program under
        # the graph's lock and keep random/shared graphs on the one-shot path.
        with self._point_lock:
            if self._point_program is not None and self._point_program[0] == signature:
                function = self._point_program[1]
            else:
                graph = graph_replace(
                    list(outputs),
                    replace={
                        self.parameters[name]: tensor
                        for name, tensor in zip(names, inputs, strict=True)
                    },
                    strict=False,
                )
                function = pm.compile(inputs, graph, mode="JAX")
                if not rvs_in_graph(graph) and not any(
                    isinstance(node, SharedVariable) for node in graph_inputs(graph)
                ):
                    self._point_program = signature, function
            return tuple(jnp.asarray(value) for value in function(*values))

    def iter_response_parameters(self, point):
        """Evaluate native response inputs once and return them in batch order."""
        nodes = tuple(node for response in self.responses for node in response.parameters)
        values = iter(self.evaluate_at(nodes, point))
        for response in self.responses:
            yield response, tuple(next(values) for _ in response.parameters)

    def evaluation(self, *, var_names=()):
        """Reuse one native clone and attach requested existing output expressions."""
        if self._evaluation is None:
            self._evaluation = self._clone()
        view = self._evaluation
        # Register optional outputs on the evaluation view only. In particular,
        # ranking-region integration must not enter the calibration NUTS target.
        pending = set(var_names) - view.model.named_vars.keys()
        if not pending:
            return view
        with view.model:
            for response in view.responses:
                batch = response.batch
                log_name, likelihood = batch.log_probability_name, batch.log_likelihood_name
                native = response.distribution
                beta = batch.likelihood_kind == "beta"
                if beta and log_name in pending:
                    raise ValueError(
                        "Beta responses expose distribution parameters, not category probabilities"
                    )
                distribution_names = batch.distribution_names
                requested_distribution = pending.intersection(distribution_names)
                rank = batch.likelihood_kind == "rank"
                if log_name in pending or requested_distribution or rank and likelihood in pending:
                    dims = (batch.obs_dim,) if beta else (batch.obs_dim, batch.category_dim)
                    if rank:
                        if log_name not in view.model.named_vars:
                            categories = np.arange(len(batch.events.labels), dtype=native.dtype)
                            pm.Deterministic(
                                log_name, pm.logp(native, categories[:, None]).T, dims=dims
                            )
                        outputs = (view.model[log_name],)
                    else:
                        node = native.owner
                        outputs = node.op.dist_params(node)
                        if log_name in pending:
                            pm.Deterministic(log_name, outputs[0], dims=dims)
                    if requested_distribution:
                        values = outputs if beta else (pt.exp(outputs[0]),)
                        for name, value in zip(distribution_names, values, strict=True):
                            if name in requested_distribution:
                                pm.Deterministic(name, value, dims=dims)
                if likelihood in pending:
                    logp = (
                        StableCategorical.logp(batch.y, view.model[log_name])
                        if rank
                        else pm.logp(native, batch.y)
                    )
                    pm.Deterministic(likelihood, logp, dims=batch.obs_dim)
        return view

    def _clone(self):
        """Clone the native graph for forward evaluation and bind marginal responses."""
        fitting_only = {node.name for response in self.responses for node in response.augmentation}
        roots = self.model.named_vars.keys() - fitting_only
        return self._rebind(_retain_outputs(self.model, roots), forward=True)

    def _rebind(self, model, *, forward=False):
        retained = {name: model[name] for name in self.parameters if name in model.named_vars}
        replacements = {self.parameters[name]: node for name, node in retained.items()}
        relocated = {
            response.distribution: model[response.batch.response_name]
            for response in self.responses
            if response.batch.response_name in model.named_vars
        }
        # Replace anonymous marginals and Fisher inputs together to retain shared expressions.
        expressions = list(
            dict.fromkeys(
                node
                for response in self.responses
                for node in (response.distribution, *(() if forward else response.parameters))
                if node not in relocated
            )
        )
        if expressions:
            updated = graph_replace(expressions, replacements, strict=False)
            relocated.update(zip(expressions, updated, strict=True))
        responses = []
        with model:
            for response in self.responses:
                batch = response.batch
                distribution = relocated[response.distribution]
                if forward and batch.response_name not in model.named_vars:
                    distribution = register_response(batch, distribution, forward=True)
                responses.append(
                    replace(
                        response,
                        parameters=()
                        if forward
                        else tuple(relocated[node] for node in response.parameters),
                        distribution=distribution,
                        augmentation=()
                        if forward
                        else tuple(model[node.name] for node in response.augmentation),
                    )
                )
        view = type(self)(
            model=model, prepared=self.prepared, parameters=retained, responses=tuple(responses)
        )
        if forward:
            view._evaluation = view
        return view


def _retain_outputs(model, names):
    """Use native graph outputs as roots, including scores without observations."""
    fgraph, nodes = fgraph_from_model(model, inlined_views=True)
    keep = {nodes[model[name]] for name in names}
    for index in reversed(range(len(fgraph.outputs))):
        if fgraph.outputs[index] not in keep:
            fgraph.remove_output(index)
    return model_from_fgraph(fgraph, mutate_fgraph=True)
