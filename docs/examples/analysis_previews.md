---
orphan: true
---

# Figure sources

Gallery images illustrate model interpretation rather than compare the quality
of different models. Each executed notebook records its own data, fit and
diagnostics. A card marked as an earlier analysis retains that figure while its
replacement workflow is being completed.

## Figures from the displayed examples

| Example | Data and figure |
| --- | --- |
| [TIRT](../tutorials/tirt_workflow.ipynb) | 120 HEXACO respondents; information across fitted Honesty–Humility and Openness scores |
| [TIRTRank](../tutorials/tirt_rank_workflow.ipynb) | 120 HEXACO respondents; observed and posterior predictive frequencies for six orders in B1 |
| [2PLM-RANK](../tutorials/two_pl_rank_workflow.ipynb) | 200 FC3 respondents; fitted endorsement curves for one HEXACO triplet |
| [GGUM-RANK](../tutorials/ggum_rank_workflow.ipynb) | 200 simulated respondents and 24 triplets; ideal-point endorsement curves with score support |
| [MGPPM](../tutorials/mgppm_workflow.ipynb) | 200 FC3 respondents; observed and replicated best–worst choices for one HEXACO triplet |
| [GTUM](../tutorials/gtum_workflow.ipynb) | 100 simulated respondents; statement locations and 95% HDIs |
| [Block and item assignments](../tutorials/joint_workflow.ipynb) | 80 simulated respondents, four triplets and four items; agency scores under shared and separate trait assignments |
| [Binary and ordinal responses](../tutorials/binary_ordinal_workflow.ipynb) | 50 simulated respondents and 12 GPCM items; information across fitted trait scores |
| [TIRT with covariates](../tutorials/tirt_covariates.ipynb) | 80 source respondents; person and item-loading regression estimates, with an explicitly generated teaching covariate |
| [JRT–TIRT](../tutorials/jrt_tirt_native.ipynb) | 100 synthetic respondents; observed and posterior-predictive response-time distributions |

| [M-TCIR](../tutorials/m_tcir_native.ipynb) | 100 synthetic respondents; matched TIRT and M-TCIR trait scores |

Diagnostics belong to the fit described in each notebook; values from earlier
simulated fits do not describe the replacement empirical analyses. The grouped
TIRT labels are teaching assignments rather than demographic groups. A shaded
score band in a statement-function plot locates the fitted scores' 5th–95th
percentiles; it is not a curve uncertainty interval.

## Reproduce the earlier figures

The [provenance record](../_static/gallery/provenance.json) describes the saved
summary inputs below. It concerns the earlier figures, including the
[statement-function illustration](../tutorials/statement_curves.ipynb), not the
new single-model fits. Download the [plotting script](../_static/gallery/render.py)
and the required CSV inputs beside it, then run `python render.py --output figures`.

- [TIRTRank variances](../_static/gallery/tirt_rank_variances.csv).
- [2PLM-RANK estimates](../_static/gallery/two_pl_loadings.csv) and
  [densities](../_static/gallery/two_pl_loading_densities.csv).
- [MGPPM intercepts](../_static/gallery/mgppm_intercepts.csv).
- [GGUM-RANK locations](../_static/gallery/ggum_rank_locations.csv) and
  [densities](../_static/gallery/ggum_rank_location_densities.csv).
- [GTUM locations](../_static/gallery/gtum_locations.csv).
- [Loading-covariate density](../_static/gallery/loading_covariate_density.csv),
  [multiplier intervals](../_static/gallery/covariate_loading_effect.csv) and
  [training predictor](../_static/gallery/covariate_loading_predictor.csv).
- [Statement curves](../_static/gallery/statement_curves_data.csv) and
  [score support](../_static/gallery/statement_curves_scores.csv).

New figures are reproduced by their notebook cells. Large fitted analysis files
stay outside the documentation.
