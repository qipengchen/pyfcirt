# Datasets

PyFCIRT bundles numerical designs and responses for the examples. Choose a
resource below, then use [load_example()](reference/generated/pyfcirt.load_example.md)
to select respondents and a measurement model.

## Empirical datasets and source files

| Study | Responses and design available | Source |
|---|---|---|
| [Detecting Careless Responding in Multidimensional Forced-Choice Questionnaires](https://doi.org/10.1177/00131644231222420) — Kupffer et al. (2024) | Big Five numerical triplet comparisons and design: 20 triplets, five traits, up to 1,169 respondents. Item wording, demographics and other subtests are excluded. | Bundled: `carelessness_big_five`. [Original files](https://github.com/rkupffer/ExploratoryAnalysesCRinMFC). |
| [Mixed-Keying or Desirability-Matching in the Construction of Forced-Choice Measures? An Empirical Investigation and Practical Recommendations](https://doi.org/10.1177/10944281241229784) — Li et al. (2025) | The bundled `autoFC` extract contains five HEXACO triplets, six traits and 541 respondents. The original study also supplies rating responses and full FC forms. | Bundled: `hexaco_triplets` (five blocks) and `hexaco_fc3` (20 blocks). Full ratings and FC3/FC4: [OSF source files](https://osf.io/yvpz3/), see [Data example](#data-example). |
| [Joint modeling of the two-alternative multidimensional forced-choice personality measurement and its response time by a Thurstonian D-diffusion item response model](https://doi.org/10.3758/s13428-019-01302-5) — Bunji and Okada (2020) | Original paired-choice responses and item design in `dat_2MFC.csv` and `items_2MFC.csv`. | Source: [OSF](https://osf.io/jswqg/). The [original-file example](howto/osf_responses.md) prepares choices only. |
| [Does multidimensional forced-choice prevent faking? Comparing the susceptibility of the multidimensional forced-choice format and the rating scale format to faking](https://doi.org/10.1037/pas0000971) — Wetzel et al. (2021) | Original forced-choice and rating-scale study files; consult the authors' design and coding information. | Source: [OSF](https://osf.io/q9uyp/). |
| [Improving reactions to forced-choice personality measures in simulated job application contexts through the satisfaction of psychological needs](https://doi.org/10.1007/s10869-023-09876-w) — Borman et al. (2024) | Original applicant-reactions study files and questionnaire information. | Source: [OSF](https://osf.io/j8wdy/). |
| [Estimating the multidimensional generalized graded unfolding model with covariates using a Bayesian approach](https://doi.org/10.3390/jintelligence11080163) — Tu et al. (2023) | Orderliness and assertiveness item ratings with age and gender. Public file item values are 1 and 2. | Source: [OSF](https://osf.io/4cr6d/). |

**Big Five.** The bundled extract follows the authors'
[rank recoding convention](https://github.com/susanne-frick/TirtAutomation/blob/master/R/retri.R).
Lower ranks are preferred. If either rank is missing, the comparison stays
missing. The extract is distributed under [CC BY 4.0](https://github.com/rkupffer/ExploratoryAnalysesCRinMFC/blob/main/License.txt).

**HEXACO.** The five-block extract retains the GPL-3.0-only license of the
[original autoFC package](https://cran.r-project.org/src/contrib/Archive/autoFC/autoFC_0.2.0.1001.tar.gz).
Use the original OSF workbooks for the full forms and Likert ratings.

## Data example

Load the prepared FC3 example from [Li et al. (2025)](https://doi.org/10.1177/10944281241229784).

```python
import pyfcirt as fc

example = fc.load_example("hexaco_fc3", model="TwoPLRank")
```

The example includes 200 first-occasion respondents, 20 triplets, and independent
statement desirability ratings. The authors gave permission to distribute this
prepared example. Each triplet records the most and least
preferred statements. The [2PLM-RANK](tutorials/two_pl_rank_workflow.ipynb) and
[MGPPM](tutorials/mgppm_workflow.ipynb) examples use these FC3 responses.
The [prior comparison](tutorials/prior_comparison.ipynb) uses the desirability ratings.

The original [OSF project](https://osf.io/yvpz3/) provides the full FC3 and FC4
forms and the 1–5 Likert ratings for both occasions. Each form contains 20
triplets of 60 statements in total. See the
[responses](https://osf.io/download/svb34/), [form designs and keying](https://osf.io/download/yw9r4/),
[desirability ratings](https://osf.io/download/ks7uq/), and
[quality-screening files](https://osf.io/download/vhm3d/).

## Simulated study data

The packaged simulations use one generating parameter set per model. Use
`include_truth=True` to retrieve the original generating values. They retain
their original parameter names and identification constraints, so match
parameters to the current model's scale before comparing estimates with truth.

The resources named `simulated_…` default to 100 respondents and include up to 500. The 50- and 100-person cohorts are subsets of the 500-person cohort.
For disjoint samples, load one cohort and split its person IDs.

| Resource | Design and response format | Use in an example |
|---|---|---|
| `simulated_tirt` | 24 triplets, four traits, binary comparisons | Simulated input for TIRT |
| `simulated_tirt_rank` | 24 triplets, four traits, full rankings | Simulated input for TIRTRank |
| `simulated_two_pl_rank` | 24 triplets, four traits, full rankings | Simulated input for 2PLM-RANK |
| `simulated_ggum_rank` | 24 triplets, four traits, full rankings | [GGUM-RANK](tutorials/ggum_rank_workflow.ipynb) |
| `simulated_gtum` | 24 triplets, four traits, binary comparisons | [GTUM](tutorials/gtum_workflow.ipynb) |
| `simulated_mgppm` | 24 triplets, four traits, full rankings | [Choosing priors](tutorials/prior_comparison.ipynb) and [Recursive scoring](tutorials/recursive_scoring.ipynb) |
| `mgppm_groups` | 120 respondents, 12 triplets, full rankings and person-level group labels (60 per group) | [Multiple groups with MGPPM](tutorials/mgppm_groups.ipynb) |
| `simulated_mirt` | Eight dichotomous items, one trait | [Binary and ordinal responses](tutorials/binary_ordinal_workflow.ipynb) |
| `simulated_mgpcm` | Twelve ordinal items, one trait | [Binary and ordinal responses](tutorials/binary_ordinal_workflow.ipynb) |
| `simulated_mggum` | Twelve ordinal items, one trait | [Binary and ordinal responses](tutorials/binary_ordinal_workflow.ipynb) |

## Other bundled teaching datasets

These datasets supply design and response tables. They have no bundled
generating parameter arrays.

| Resource | Stored sample and input | Where it is used |
|---|---|---|
| `tirt_study` | 100 synthetic respondents, 24 triplets, binary comparisons | [Quick start](guide/getting_started.md) |
| `mixed_measurement` | 80 synthetic respondents, four triplets and four individual items | [Models by block and item](tutorials/joint_workflow.ipynb) |
| `tirt_dif` | 800 synthetic respondents in two groups, 12 dyads and known loading and cutpoint differences | [TIRT DIF in R](tutorials/tirt_dif_r.ipynb) |
| `custom_tirt` | 100 synthetic respondents, nine dyads, person and statement predictors | [Loading companion tables](reference/generated/pyfcirt.load_example.md) |
| `jrt_tirt` | 100 synthetic respondents, nine dyads, matched response times | [JRT–TIRT](tutorials/jrt_tirt_native.ipynb) and [M-TCIR](tutorials/m_tcir_native.ipynb) |

Use [simulate()](reference/generated/pyfcirt.simulate.md) to generate new
responses from a different design or parameter set.
