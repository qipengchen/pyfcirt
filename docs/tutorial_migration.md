# Adapt an existing forced-choice analysis

Start with the [shared workflow](tutorials/tirt_workflow.ipynb): declare the full design, prepare
responses, inspect native priors, check prior predictions, fit, inspect sampling
diagnostics, evaluate predictions and score new respondents. Model-specific
arrays belong in adapters rather than in the analysis notebook.

## Preserve the observed event

Retain person, block and statement identifiers, presentation order, keying,
location sides, collection response formats and response scales. A preference list such as
`[1, 2, 3]` records positions from most to least preferred. It is not a vector of
statement ratings. Declare the full category scale even when some categories do
not occur in the observed sample.

Use [normalized tables](reference/generated/pyfcirt.ResponseTable.md) and the [response-file examples](howto/read_responses.md).
Preserve missing answers separately from structural PICK/MOLE omissions. Never
complete a partial ranking or remove cycles to make a ranking model accept it.
TIRT and GTUM retain comparison likelihoods; decomposing a larger ranking into
pairs does not create a joint ranking distribution. See
[comparison laws](models/comparisons.md) and [pair projections](pair_projection.md).

## Match the model before comparing results

Check likelihoods, priors, identification constraints, transformations, coding and
observation units. Old notebook output does not validate revised defaults.
Compare uncertainty and diagnostics as well as means. Backend changes do not
resolve identification or a mismatch in the observation law.

Use [custom priors](reference/generated/pyfcirt.Model.set_priors.md) for declarative analyses and the public
[measurement distribution classes](api.md) inside a caller-owned PyMC graph for custom
parameters. Retain the construction code for arbitrary PyMC expressions.
[Mixed blocks](mixed_models.md) and [joint FC–Likert models](joint_models.md)
are experimental and have explicit family and sharing restrictions.

## Save current analyses

[Save and load](reference/results.md) the current analysis state with native xarray and explicit sample attachment.
Only the current result format is supported; old archives are rejected.
Keep the environment and source for older analyses rather than relabelling their
manifest. Notebook figures and bare arrays are not reconstructible package fits.
Scoring new people holds a validated calibration point fixed and excludes
calibration uncertainty.

Preserve the provenance and redistribution terms of external datasets and code.
Package licensing does not replace third-party permissions. An extension discussed
in a paper is not necessarily implemented; check the [capability boundaries](roadmap.md).
