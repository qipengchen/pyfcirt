"""Independent scalar Thurstonian interval probabilities."""

import numpy as np
from scipy.integrate import quad
from scipy.special import log_expit, log_ndtr
from scipy.stats import norm


def tirt_log_probs(eta, cuts, link):
    """Scalar interval integration, scaled to avoid underflow in normal tails."""
    z = np.asarray(cuts) - eta
    if link == "logit":
        z = 1.702 * z
        logcdf = log_expit

        def logdensity(x):
            return float(log_expit(x) + log_expit(-x))
    else:
        logcdf = log_ndtr
        logdensity = norm.logpdf
    result = [float(logcdf(z[0]))]
    for lower, upper in zip(z[:-1], z[1:], strict=True):
        maximum = logdensity(np.clip(0, lower, upper))
        area = quad(
            lambda x, maximum=maximum: np.exp(logdensity(x) - maximum),
            lower,
            upper,
            epsabs=1e-13,
            epsrel=1e-12,
        )[0]
        result.append(np.log(area) + maximum)
    result.append(float(logcdf(-z[-1])))
    return np.array(result)


def gtum_log_probs(eta, cutpoints):
    """Direct algebraic logistic interval formula, independent of reflected CDFs."""
    z = np.asarray(cutpoints) - eta
    result = [-np.logaddexp(0, -z[0])]
    for lower, upper in zip(z[:-1], z[1:], strict=True):
        result.append(
            upper
            + np.log(-np.expm1(lower - upper))
            - np.logaddexp(0, upper)
            - np.logaddexp(0, lower)
        )
    result.append(-np.logaddexp(0, z[-1]))
    return np.array(result)
