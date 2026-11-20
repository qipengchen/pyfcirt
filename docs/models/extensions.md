# Additional model families

The supported analyses are described in the [model guides](index.md).
BRB IRT, joint response-time TIRT, Thurstonian diffusion models and other
specialized FC extensions are not available through the standard model API.
Their likelihoods and identification constraints cannot be inferred from a
shared TIRT label.

For custom research models, the [external PyMC interface](../reference/generated/pyfcirt.Regression.md)
accepts caller-owned latent traits and parameter expressions where the
component likelihood is supported. Retain the construction code and validate
the full statistical model; a custom graph does not automatically inherit
standard prediction, scoring or persistence support.
