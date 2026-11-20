"""Bayesian forced-choice analysis with native PyMC graphs and JAX kernels."""

# Apply precision defaults before numerical model components are imported.
from pyfcirt.config import set_platform

# isort: split

from pyfcirt.analysis.checks import chi_ppp, residuals
from pyfcirt.analysis.comparison import dic, waic
from pyfcirt.analysis.information import information
from pyfcirt.analysis.output import parameter_point
from pyfcirt.analysis.prediction import (
    log_likelihood,
    posterior_predictive,
    predictive_view,
    prior_predictive,
)
from pyfcirt.analysis.reliability import (
    empirical_reliability,
    information_reliability,
    reliability,
)
from pyfcirt.analysis.summary import coef, data_summary, response_data, response_summary
from pyfcirt.analysis.traits import trait_summary
from pyfcirt.covariates.specification import Regression, RegressionOptions
from pyfcirt.data import (
    BlockSpec,
    CategoryScale,
    ComparisonColumn,
    ExampleData,
    FCDesignMapping,
    FCInput,
    ItemDesignMapping,
    ItemInput,
    ItemOverride,
    PreparedInputs,
    ResponseTable,
    UnitIntervalScale,
    generation_schedule,
    load_example,
)
from pyfcirt.inference import sample_gtum
from pyfcirt.model import Model
from pyfcirt.models import (
    GGUM,
    GPCM,
    GTUM,
    MGPPM,
    TIRT,
    DichotomousIRT,
    GGUMRank,
    IdentificationWarning,
    PopulationSpec,
    TIRTRank,
    TwoPLRank,
)
from pyfcirt.plotting import plot_ppp, plot_response_distribution, plot_ridge
from pyfcirt.scoring import score, score_model
from pyfcirt.scoring.approximation import PosteriorApproximation
from pyfcirt.simulation import simulate

__version__ = "0.1.0.dev0"

__all__ = [
    "set_platform",
    "Model",
    "PreparedInputs",
    "Regression",
    "RegressionOptions",
    "IdentificationWarning",
    "PopulationSpec",
    "PosteriorApproximation",
    "MGPPM",
    "TwoPLRank",
    "GGUMRank",
    "TIRT",
    "GTUM",
    "TIRTRank",
    "DichotomousIRT",
    "GPCM",
    "GGUM",
    "BlockSpec",
    "CategoryScale",
    "ComparisonColumn",
    "FCDesignMapping",
    "FCInput",
    "ExampleData",
    "ItemDesignMapping",
    "ItemInput",
    "ItemOverride",
    "ResponseTable",
    "UnitIntervalScale",
    "generation_schedule",
    "load_example",
    "data_summary",
    "response_data",
    "response_summary",
    "coef",
    "trait_summary",
    "parameter_point",
    "posterior_predictive",
    "prior_predictive",
    "predictive_view",
    "log_likelihood",
    "chi_ppp",
    "residuals",
    "information",
    "reliability",
    "empirical_reliability",
    "information_reliability",
    "waic",
    "dic",
    "plot_ppp",
    "plot_response_distribution",
    "plot_ridge",
    "score",
    "score_model",
    "sample_gtum",
    "simulate",
]
