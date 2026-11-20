"""Independent subjective-state single-item equations."""

import numpy as np
from scipy.special import logsumexp


def subjective_log_probs(theta, loading, location, threshold):
    """Enumerate all 2K subjective PCM states before collapsing mirrored states."""
    free_steps = list(threshold[1:])
    steps = [0.0, *free_steps, 0.0, *[-value for value in reversed(free_steps)]]
    state_logs = [
        loading * (state * (theta - location) - sum(steps[: state + 1]))
        for state in range(len(steps))
    ]
    denominator = logsumexp(state_logs)
    return np.array(
        [
            np.logaddexp(state_logs[k], state_logs[-1 - k]) - denominator
            for k in range(len(threshold))
        ]
    )


def native_category_row(point, owner, item, suffix):
    """Read the declaration-local category array by its native item coordinate."""
    for name, array in point.data_vars.items():
        if name.startswith(f"{owner['id']}::") and name.endswith(suffix):
            dimension = array.dims[0]
            if item in array.coords[dimension].values:
                return array.sel({dimension: item}).values
    raise AssertionError("Missing native item category parameter")
