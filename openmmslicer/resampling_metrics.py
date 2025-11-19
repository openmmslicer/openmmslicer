import numpy as _np
from scipy.stats import entropy as _entropy


class EffectiveSampleSize:
    """
    Calculates the effective sample size of one divided by the second moment of the normalised weights.
    Goes from 1/n_weights to 1.
    """
    def __init__(self, ensemble=None, defaultValue=0.5, defaultTol=0.01):
        self.ensemble = ensemble
        self.defaultValue = defaultValue
        self.defaultTol = defaultTol

    def __call__(cls, weights):
        """
        Evaluates the metric.

        Parameters
        ----------
        weights : list or numpy.ndarray
            The input weights.

        Returns
        -------
        metric : float
            The weight-dependent metric.
        """
        weights = _np.asarray(weights, dtype=_np.float32)
        return _np.sum(weights) ** 2 / _np.sum(weights ** 2) / weights.shape[0]


class ExpectedSampleSize(EffectiveSampleSize):
    """
    Calculates the expected sample size based on weight probabilities. Goes from 1/n_weights to 1.
    """
    def __call__(cls, weights):
        weights = _np.asarray(weights, dtype=_np.float32)
        weights /= _np.sum(weights)
        weights *= weights.shape[0]
        weights[weights > 1] = 1
        return _np.average(weights)


class WorstCaseSampleSize(EffectiveSampleSize):
    """
    This metric denotes the inverse of the maximum weight. Goes from 1/n_weights to 1.
    """
    def __call__(cls, weights):
        weights = _np.asarray(weights, dtype=_np.float32)
        return _np.sum(weights) / _np.max(weights) / weights.shape[0]


class WorstCaseSystematicSampleSize(EffectiveSampleSize):
    """
    If systematic resampling is used, this metric calculates the samples which are certain to be resampled and returns
    the number of unique certainly resampled samples. Goes from 1/n_weights to 1.
    """
    def __call__(cls, weights):
        weights = _np.asarray(weights, dtype=_np.float32)
        int_n_walkers = _np.floor(weights.shape[0] * weights)
        return int_n_walkers[int_n_walkers >= 1].shape[0] / weights.shape[0]


class ExpWeightEntropy(EffectiveSampleSize):
    """
    Calculates the exponential of the entropy of the normalised weights. Goes from 1/n_weights to 1.
    """
    def __call__(cls, weights):
        weights = _np.asarray(weights, dtype=_np.float32)
        weights /= _np.sum(weights)
        return _np.exp(_entropy(weights)) / weights.shape[0]
