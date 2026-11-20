# PyFCIRT

Bayesian forced-choice modeling in Python.

[![Checks](https://github.com/qipengchen/pyfcirt/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/qipengchen/pyfcirt/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/qipengchen/pyfcirt/branch/main/graph/badge.svg)](https://codecov.io/gh/qipengchen/pyfcirt)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](https://qipengchen.github.io/pyfcirt/installation.html)
[![License](https://img.shields.io/badge/License-MIT%20%26%20Apache--2.0-blue)](NOTICE)

<!-- quick-start-start -->

## Overview

PyFCIRT is a high-level Bayesian forced-choice modeling interface written in
Python. Built on the [PyMC](https://github.com/pymc-devs/pymc) probabilistic
programming framework, it is designed to make forced-choice models easy to
specify, fit, and evaluate. GPU-accelerated sampling is available through JAX.

## Supported models

**Forced-choice: dominance models**

- Thurstonian item response theory model (`TIRT`).
- Thurstonian IRT model with a joint-ranking likelihood (`TIRTRank`).
- Two-parameter logistic model for rankings (`TwoPLRank`).
- Multidimensional generalized partial preference model (`MGPPM`).

**Forced-choice: ideal-point models**

- Generalized Thurstonian unfolding model (`GTUM`).
- Generalized graded unfolding model for rankings (`GGUMRank`).

**Likert and dichotomous models**

- One- to four-parameter logistic IRT models (`DichotomousIRT`).
- Generalized partial credit model (`GPCM`).
- Generalized graded unfolding model (`GGUM`).

Different measurement models can be combined within a test, including
forced-choice, Likert, and dichotomous models.

## Installation

Use Python 3.12 or newer. Install from GitHub:

```console
python -m pip install "pyfcirt[plots,io,covariates] @ https://github.com/qipengchen/pyfcirt/archive/refs/heads/main.zip"
```

This includes plotting, NetCDF saving and covariate formulas.

## Your first analysis

Fit TIRT to a simulated teaching dataset with 100 respondents and 24 triplets.
The design table assigns each statement to a block and trait; the response table
contains one person's judgment of a named statement pair per row.

```python
import pyfcirt as fc

example = fc.load_example("tirt_study", model="TIRT")
design_df = example.data.design
response_df = example.data.responses.frame

# design_df.head(3):
# block_id position statement_id trait keying
#       B1        1           S1    T1      1
#       B1        2           S2    T2      1
#       B1        3           S3    T3      1

# response_df.head(3):
# person_id block_id statement_1 statement_2 response
#        P1       B1          S1          S2        0
#        P1       B1          S1          S3        1
#        P1       B1          S2          S3        1
```

Register the design and responses, then fit TIRT.

```python
model = fc.Model()
model.add_fc(
    design_df,
    response_df,
    format="comparison_long",
    model="TIRT",
)
model.prepare()
model.fit(random_seed=42)
model.coef(blocks=["B1"], kind="stats", round_to="auto")
```

`model="TIRT"` selects the measurement model. Each response records a pair
judgment. Response `1` prefers the first statement;
`0` prefers the second. In the design, `position` gives presentation order and
`keying` gives the loading direction on the named trait.

`model.coef()` reports posterior means, standard deviations and uncertainty
intervals for the statements and pairs in B1.

Follow the [examples](https://qipengchen.github.io/pyfcirt/examples.html) for
prior prediction, diagnostics and model checks, and use the
[API reference](https://qipengchen.github.io/pyfcirt/api.html) for individual calls.
Use [Prepare your data](https://qipengchen.github.io/pyfcirt/reference/generated/pyfcirt.ResponseTable.html) to define
your own design and read CSV or spreadsheet responses.

<!-- quick-start-end -->

## Documentation

- [Quick start](https://qipengchen.github.io/pyfcirt/guide/getting_started.html)
- [Installation and GPU support](https://qipengchen.github.io/pyfcirt/installation.html)
- [API reference](https://qipengchen.github.io/pyfcirt/api.html)
- [Examples](https://qipengchen.github.io/pyfcirt/examples.html)

## Contributing

Questions, bug reports, and feature suggestions are welcome through
[GitHub Issues](https://github.com/qipengchen/pyfcirt/issues).

For bug reports, include a small reproducible example and the package versions
used. Contributions can be submitted through pull requests.

## License

PyFCIRT is primarily licensed under the [MIT License](LICENSE).
See [NOTICE](NOTICE) for the JAX attribution retained from earlier revisions.
