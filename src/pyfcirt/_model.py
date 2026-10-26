"""Small containers joining a native graph, statistical metadata and original data."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from .data import FCData
from .models.mgppm._data import MGPPMInputs

if TYPE_CHECKING:
    import pymc as pm
    import xarray as xr


@dataclass(frozen=True)
class ModelSpec:
    """Serializable model definition; contains no executable graph or storage path."""

    model_id: str
    model_version: str
    settings: dict
    components: list[dict]
    observation_map: list[dict]
    variable_roles: dict
    observation_unit: str
    likelihood_kind: str
    data_fingerprint: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class BuiltModel:
    """A native PyMC graph with its data, declared statistical target and arrays."""

    model: pm.Model
    spec: ModelSpec
    data: FCData
    inputs: MGPPMInputs

    def fit(self, **kwargs):
        """Delegate posterior fitting to the common sampling implementation."""
        from .inference import fit

        return fit(self, **kwargs)

    def prior_predictive(self, *, draws: int = 500, random_seed: int | None = None) -> xr.DataTree:
        """Generate prior parameters and responses without posterior fitting."""
        from .predictive import prior_predictive

        return prior_predictive(self, draws=draws, random_seed=random_seed)

    def response_frame(
        self, draws: xr.DataTree, *, group: str = "prior_predictive", chain: int = 0, draw: int = 0
    ):
        """Decode one replicated dataset to the original block-response contract."""
        from .predictive import response_frame

        return response_frame(self, draws, group=group, chain=chain, draw=draw)
