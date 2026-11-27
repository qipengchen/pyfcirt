"""Point-zero DIF priors lowered to native PyMC indicator and slab nodes."""

import numpy as np
import pymc as pm
from pytensor.tensor.random.op import RandomVariable


class SpikeAndSlabRV(RandomVariable):
    """Scalar prior declaration for a zero point mass and a Normal slab."""

    name = "spike_and_slab"
    signature = "(),()->()"
    dtype = "floatX"
    _print_name = ("SpikeAndSlab", r"\operatorname{SpikeAndSlab}")

    @classmethod
    def rng_fn(cls, rng, inclusion_prob, slab_sd, size):
        slab = rng.normal(0, slab_sd, size=size)
        return slab * (rng.random(np.shape(slab)) < inclusion_prob)

    @staticmethod
    def bind(prior, *, name, shape, dims):
        probability, scale = prior.owner.op.dist_params(prior.owner)
        included = pm.Bernoulli(f"{name}_included", p=probability, shape=shape, dims=dims)
        slab = pm.Normal(f"{name}_slab", sigma=scale, shape=shape, dims=dims)
        return pm.Deterministic(name, included * slab, dims=dims)


spike_and_slab = SpikeAndSlabRV()
