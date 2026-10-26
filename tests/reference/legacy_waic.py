"""Run only in the isolated ArviZ 0.22.0 reference environment."""

import hashlib
import importlib.metadata
import json
import sys
import warnings
from pathlib import Path

import arviz as az
import arviz.stats.stats as stats_module
import numpy as np

assert az.__version__ == "0.22.0", az.__version__
source = json.loads(Path(sys.argv[1]).read_text(encoding="utf8"))
idata = az.from_dict(
    log_likelihood={"response": np.asarray(source["values"], dtype=np.float64)},
    coords={"obs_id": source["obs_id"]},
    dims={"response": ["obs_id"]},
)
output = {}
for scale in ["log", "deviance"]:
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        result = az.waic(idata, pointwise=True, var_name="response", scale=scale)
    output[scale] = {
        "elpd_waic": float(result.elpd_waic),
        "se": float(result.se),
        "p_waic": float(result.p_waic),
        "n_samples": int(result.n_samples),
        "n_data_points": int(result.n_data_points),
        "warning": bool(result.warning),
        "pointwise": result.waic_i.values.tolist(),
        "scale": result.scale,
        "warnings": [str(w.message) for w in captured],
    }
output["versions"] = {
    name: importlib.metadata.version(name)
    for name in ["arviz", "numpy", "scipy", "xarray", "pandas"]
}
output["reference_source_sha256"] = hashlib.sha256(
    Path(stats_module.__file__).read_bytes()
).hexdigest()
Path(sys.argv[2]).write_text(json.dumps(output, indent=2), encoding="utf8")
