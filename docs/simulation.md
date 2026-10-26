# Simulating MGPPM responses

`simulate_mgppm` generates responses at explicit parameter values. It neither
draws priors nor estimates parameters. Supply a validated `FCDesign`, one row
per person in `theta`, and statement-indexed parameter series:

```python
import pandas as pd
from pyfcirt import FCDesign
from pyfcirt.simulation import simulate_mgppm

design = FCDesign.from_mapping(
    traits={"T1": ["A"], "T2": ["B"]},
    blocks={"pair": ["A", "B"]},
)
data = simulate_mgppm(
    design,
    theta=pd.DataFrame({"T1": [0.2], "T2": [-0.3]}, index=["P1"]),
    loading_magnitude=pd.Series({"A": 1.0, "B": 0.8}),
    intercept=pd.Series({"A": 0.1, "B": -0.1}),
    random_seed=42,
)
print(data.to_frame())
```

Parameter labels are aligned explicitly. Loading magnitudes must be positive,
all values must be finite, and ranking intercepts must sum to zero within each
block. Statement keying determines signed loadings. Graded intercepts have no
zero-sum constraint. Supported event types and block sizes follow the MGPPM
adapter; PICK and MOLE retain their structural missing ranks.

This function generates every supplied person-block pair and returns `FCData`.
It does not simulate an administration or missing-response process. Use
`Model.prior_predictive` for generation under the configured model priors.
