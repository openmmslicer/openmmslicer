import itertools as _it
import random as _random

import numpy as _np


class MultinomialResampler:
    """
    A basic multinomial resampler which resamples each sample independently of the others with replacement.
    """
    @staticmethod
    def resample(samples, weights=None, n_walkers=None, n_samples=1):
        """
        Resamples samples multinomially based on given weights.

        Parameters
        ----------
        samples : list
            The input samples.
        weights : list
            The weights associated with the samples. Default is equal weights.
        n_walkers: int
            The number of resampled samples per batch. Default is the length of the samples.
        n_samples: int
            The number of resample batches. Default is one.

        Returns
        -------
        resamples : [list]
            A list of lists, containing the resampled samples.
        """
        if weights is not None:
            weights = _np.asarray(weights)
            weights /= sum(weights)
        else:
            weights = _np.repeat(1 / len(samples), len(samples))

        if n_walkers is None:
            n_walkers = len(weights)

        return [_random.choices(samples, weights, k=n_walkers) for _ in range(n_samples)]


class SystematicResampler:
    """
    The most conservative resampler, which preserves as many samples as possible. Based on the method in:
    http://dx.doi.org/10.3150/12-BEJSP07. An additional review can be found in:
    http://people.isy.liu.se/rt/schon/Publications/HolSG2006.pdf.
    """
    @staticmethod
    def resample(samples, weights=None, n_walkers=None, n_samples=1):
        """
        Resamples samples systematically based on given weights.

        Parameters
        ----------
        samples : list
            The input samples.
        weights : list
            The weights associated with the samples. Default is equal weights.
        n_walkers: int
            The number of resampled samples per batch. Default is the length of the samples.
        n_samples: int
            The number of resample batches. Default is one.

        Returns
        -------
        resamples : [list]
            A list of lists, containing the resampled samples.
        """
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
