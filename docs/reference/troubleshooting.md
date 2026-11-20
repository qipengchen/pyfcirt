# Troubleshooting

## Python cannot import PyFCIRT

Import `sys` and call `print(sys.executable)` to check the active interpreter. Install with
that interpreter and select the same notebook kernel. Follow
[Installation](../guide/getting_started.md#installation).

## A response or ID is rejected

Find the row reported in the error. Compare its columns, labels and categories
with the declared [response storage format](generated/pyfcirt.ResponseTable.md).
For example, `"01"` and `1` are distinct IDs, and the order cell
`[3, 1, 2]` places presentation position 3 first.

## A predictor table cannot be aligned

Give each used entity one row with the same ID as in the design. Check that
used columns are complete. For scoring, supply the training formula's
predictors with compatible categorical levels.
See [covariates](generated/pyfcirt.Regression.md).

## The first fit or prediction is slow

PyMC may build and compile a computation graph on the first call. Later calls
can reuse it. See the
[estimation backends](../examples/estimation_backends.md) before changing devices.

## Chains disagree or diverge

Locate the affected parameters using rank plots, R-hat and effective sample
size. Check response coding, identifying constraints and prior predictions.
See [the TIRT workflow](../tutorials/tirt_workflow.ipynb) for the diagnostic steps.

## A GPU or sampler is unavailable

Install the chosen sampler in the active environment. For GPU sampling,
check the JAX backend and its driver. See [GPU setup](../examples/gpu.md).

## Saved samples do not attach

Recreate the original responses, design, priors, regressions and
parameterization. Inspect both the data and sample labels, including any
fixed calibration snapshot. For responses from different people, prepare
[a scoring model](generated/pyfcirt.score_model.md).
See [saving and restoration](results.md).
