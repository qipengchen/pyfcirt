# Measurement models

Declare a response law by its canonical name or a configuration object. [Native distribution classes](distributions.md) use the same laws in caller-owned PyMC graphs.

## Choose a response law

Three choices describe an FC model: how it represents a block, the response
process it assumes, and the response format used in the questionnaire. Dominance
models describe monotonic endorsement or utility; ideal-point (unfolding) models
allow endorsement or utility to peak near a statement's location.

| Model name | Block decomposition | Response process | Test / response format |
| --- | --- | --- | --- |
| [TIRT](../models/tirt.md) | Pairwise | Dominance | RANK, PICK, MOLE; binary, graded or continuous pair ratings |
| [TIRTRank](../models/tirt_rank.md) | Pairwise | Dominance | RANK, PICK, MOLE |
| [2PLM-RANK](../models/two_pl_rank.md) (`TwoPLRank`) | Sequential | Dominance | RANK, PICK, MOLE |
| [GGUM-RANK](../models/ggum_rank.md) (`GGUMRank`) | Sequential | Ideal point | RANK, PICK, MOLE |
| [GTUM](../models/gtum.md) | Pairwise | Ideal point | RANK, PICK, MOLE; binary, graded or continuous pair ratings |
| [MGPPM](../models/mgppm.md) | Ranking pattern | Dominance | RANK, PICK, MOLE; graded ratings for two-statement blocks |
| [Dichotomous IRT](../models/single_items.md#dichotomous-irt) (1PL–4PL) | — | Dominance | Binary item responses |
| [GPCM](../models/single_items.md#gpcm) | — | Dominance | Ordered item categories |
| [GGUM](../models/single_items.md#ggum) | — | Ideal point | Ordered item categories |

The sequential-selection, pairwise-comparison and ranking-pattern distinction
follows [Furr and Fu (2025)](https://doi.org/10.1017/psy.2025.10054).
Block decomposition applies to FC blocks; individual-item models have no such
decomposition. TIRTRank evaluates the Gaussian comparison relations jointly. TIRT and GTUM use
a composite likelihood when ranking-derived comparisons are multiplied; their
pair factors are not a joint ranking probability. Continuous pair ratings use
their Beta links. These are the formats supported by the package; model-specific
notes give the likelihoods and assumptions.


## Forced-choice: dominance

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   ~pyfcirt.TIRT
   ~pyfcirt.TIRTRank
   ~pyfcirt.TwoPLRank
   ~pyfcirt.MGPPM
```

## Forced-choice: ideal point

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   ~pyfcirt.GGUMRank
   ~pyfcirt.GTUM
```

## Binary and ordinal responses

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   ~pyfcirt.DichotomousIRT
   ~pyfcirt.GPCM
   ~pyfcirt.GGUM
```

## Identification

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   ~pyfcirt.IdentificationWarning
```

```{toctree}
:hidden:

../models/index
```
