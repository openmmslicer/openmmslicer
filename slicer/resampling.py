import itertools as _it
import random as _random

import numpy as _np


class MultinomialResampler:
    @staticmethod
    def resample(samples, weights=None, n_walkers=None, n_samples=1):
        if weights is not None:
            weights = _np.asarray(weights)
            weights /= sum(weights)
        else:
            weights = _np.repeat(1 / len(samples), len(samples))

        if n_walkers is None:
            n_walkers = len(weights)

        return [_random.choices(samples, weights, k=n_walkers) for _ in range(n_samples)]


class SystematicResampler:
    @staticmethod
    def resample(samples, weights=None, n_walkers=None, n_samples=1):
        # this algorithm is based on the method in: http://dx.doi.org/10.3150/12-BEJSP07
        # additional review: http://people.isy.liu.se/rt/schon/Publications/HolSG2006.pdf
        if weights is not None:
            weights = _np.asarray(weights)
            weights /= sum(weights)
        else:
            weights = _np.repeat(1 / len(samples), len(samples))

        if n_walkers is None:
            n_walkers = len(weights)

        cdf = _np.array([0] + list(_it.accumulate(weights)))
        randnums = _np.random.uniform(size=(n_samples, 1)) / n_walkers
        rational_weights = _np.linspace(0, 1, endpoint=False, num=n_walkers)
        all_cdf_points = randnums + rational_weights

        int_weights = [_np.histogram(cdf_points, bins=cdf)[0] for cdf_points in all_cdf_points]
        all_samples = [sum([i * [x] for i, x in zip(int_weight, samples)], []) for int_weight in int_weights]

        return all_samples
