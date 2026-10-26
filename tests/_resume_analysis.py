"""Execute saved-analysis operations in a fresh installed-package process."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import pyfcirt as fc

saved = fc.load(sys.argv[1])
fitted = saved.rebuild()
target = Path(sys.argv[2])
target.mkdir()
np.save(target / "logp.npy", fitted.log_likelihood(progressbar=False).values)
replicated = fitted.posterior_predictive(random_seed=101, progressbar=False)
frame = fitted.response_frame(replicated)
(target / "predicted.json").write_text(json.dumps(frame.to_dict("records")), encoding="utf8")
empty = pd.DataFrame(columns=["person_id", "block_id", "response"])
admin = pd.DataFrame(
    {
        "person_id": ["fresh-new"],
        "block_id": [fitted.built.inputs.block_ids[0]],
        "status": ["missing"],
    }
)
new_data = fc.FCData.from_frames(empty, fitted.built.data.design, administration=admin)
scored = fitted.score(new_data, draws=4, chains=1, random_seed=102)
np.save(target / "scores.npy", scored.samples.values)
