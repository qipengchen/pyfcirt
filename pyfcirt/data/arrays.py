"""Collect arrays on validated axes before constructing the labelled dataset."""

import numpy as np
import xarray as xr


def freeze_arrays(dataset):
    """Make owned preparation buffers safe to share with xarray results."""
    indexes = dataset.indexes
    for name, variable in dataset.variables.items():
        # IndexVariable.data can materialize a cast of the whole label column.
        values = indexes[name].values if name in indexes else variable.data
        if isinstance(values, np.ndarray):
            values.flags.writeable = False


class LayoutBuilder:
    """Own preparation arrays without repeatedly aligning an expanding Dataset.

    Input tables have already established label order. All writes here use those
    integer axes; xarray receives the complete variables and coordinates once.
    This object is temporary and is never part of the saved analysis state.
    """

    def __init__(self, *, coords, attrs):
        self.variables = {}
        self.sizes = {}
        self.attrs = dict(attrs)
        self._coordinates = {}
        self.coords = self._coordinates.keys()
        self.add_coords(coords)

    def add_coords(self, coordinates):
        for key, value in coordinates.items():
            self[key] = value
        self._coordinates.update(dict.fromkeys(coordinates))

    def __setitem__(self, key, value):
        if isinstance(value, xr.DataArray):
            value = value.variable
        elif not isinstance(value, xr.Variable):
            value = (
                xr.Variable(*value)
                if isinstance(value, tuple)
                else xr.Variable((key,), np.asarray(value))
            )
        for dimension, size in zip(value.dims, value.shape, strict=True):
            if dimension in self.sizes and self.sizes[dimension] != size:
                raise ValueError(f"Conflicting preparation dimension {dimension!r}")
            self.sizes[dimension] = size
        self.variables[key] = value

    def __getitem__(self, key):
        return self.variables[key]

    def __getattr__(self, key):
        try:
            return self.variables[key]
        except KeyError:
            raise AttributeError(key) from None

    def update(self, dataset):
        self.add_coords({key: dataset.variables[key] for key in dataset.coords})
        for key in dataset.data_vars:
            self[key] = dataset.variables[key]

    def build(self):
        return xr.Dataset(
            {k: v for k, v in self.variables.items() if k not in self.coords},
            coords={k: self.variables[k] for k in self.coords},
            attrs=self.attrs,
        )


def merge_prepared(datasets):
    """Combine aligned datasets and builders in one xarray construction."""
    variables, coordinates = {}, {}
    for dataset in datasets:
        for name, value in dataset.variables.items():
            if name in variables and not variables[name].identical(value):
                raise ValueError(f"Conflicting prepared variable {name!r}")
            variables.setdefault(name, value)
        coordinates.update(dict.fromkeys(dataset.coords))
    return xr.Dataset(
        {name: value for name, value in variables.items() if name not in coordinates},
        coords={name: variables[name] for name in coordinates},
        attrs=datasets[0].attrs,
    )
