"""Batched score-weighted categorical and partial-event MGPPM probabilities."""

import numpy as np
import pytensor.tensor as pt
from pytensor.tensor.variable import TensorVariable

from ._data import MGPPMGroup, MGPPMInputs


def statement_utilities(
    theta: TensorVariable,
    loading_magnitude: TensorVariable,
    intercept: TensorVariable,
    inputs: MGPPMInputs,
    group: MGPPMGroup,
) -> TensorVariable:
    """Gather shared parameters once per effective person/block event."""
    statement = group.statement_index[group.block_index]
    trait = inputs.statement_trait[statement]
    signed_loading = pt.as_tensor_variable(loading_magnitude) * inputs.keying
    return (
        signed_loading[statement] * pt.as_tensor_variable(theta)[group.person_index[:, None], trait]
        - pt.as_tensor_variable(intercept)[statement]
    )


def event_log_probs(
    utilities: TensorVariable, scores: np.ndarray, membership: np.ndarray
) -> TensorVariable:
    """Return normalized event log probabilities along the last axis.

    Leading batch axes are preserved. Membership is an event-by-pattern Boolean
    partition; generated and validated by the model's explicit data adapter.
    No observation loops, probability clipping or missing-value imputation occur.
    """
    logits = pt.matmul(pt.as_tensor_variable(utilities), pt.as_tensor_variable(scores).T)
    pattern_logp = logits - pt.logsumexp(logits, axis=-1, keepdims=True)
    if membership.shape[0] == membership.shape[1] and np.array_equal(
        membership, np.eye(membership.shape[0], dtype=bool)
    ):
        return pattern_logp
    return pt.logsumexp(pt.where(membership, pattern_logp[..., None, :], -np.inf), axis=-1)


def numpy_event_log_probs(
    utilities: np.ndarray, scores: np.ndarray, membership: np.ndarray
) -> np.ndarray:
    """Evaluate the same event target for fixed-parameter generation and summaries."""
    logits = np.asarray(utilities, dtype=float) @ scores.T
    if not np.isfinite(logits).all():
        raise ValueError("MGPPM utilities must produce finite logits")
    pattern_logp = logits - np.logaddexp.reduce(logits, axis=-1, keepdims=True)
    if membership.shape[0] == membership.shape[1] and np.array_equal(
        membership, np.eye(membership.shape[0], dtype=bool)
    ):
        return pattern_logp
    return np.logaddexp.reduce(np.where(membership, pattern_logp[..., None, :], -np.inf), axis=-1)
