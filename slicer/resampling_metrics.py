import numpy as _np
from scipy.stats import entropy as _entropy


class EffectiveSampleSize:
    """
    Calculates the effective sample size of one divided by the second moment of the normalised weights. Goes from 1 to
    n_weights.
    """
    @classmethod
    def defaultValue(cls, n_weights):
        """float: The default value of the metric, dependent on the number of weights. Default is n_weights / 5"""
        return n_weights / 5

    @classmethod
    def defaultTol(cls, n_weights):
        """float: The default tolerance of the metric, dependent on the number of weights. Default is 0.1 / n_weights"""
        return 0.1 / n_weights

    @classmethod
    def evaluate(cls, weights):
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
        return _np.sum(weights) ** 2 / _np.sum(weights ** 2)

    def serialise(self):
        return self.__dict__


class ExpectedSampleSize(EffectiveSampleSize):
    """
    Calculates the expected sample size based on weight probabilities. Goes from 1 to n_weights.
    """
    @classmethod
    def evaluate(cls, weights):
        weights = _np.asarray(weights, dtype=_np.float32)
        weights /= _np.sum(weights)
        n_walkers = weights.shape[0]
        weights *= n_walkers
        weights = weights[weights < 1]
        return n_walkers - (weights.shape[0] - _np.sum(weights))


class WorstCaseSampleSize(EffectiveSampleSize):
    """
    This metric denotes the inverse of the maximum weight. Goes from 1 to n_weights.
    """
    @classmethod
    def evaluate(cls, weights):
        weights = _np.asarray(weights, dtype=_np.float32)
        return _np.sum(weights) / _np.max(weights)


class WorstCaseSystematicSampleSize(EffectiveSampleSize):
    """
    If systematic resampling is used, this metric calculates the samples which are certain to be resampled and returns
    the number of unique certainly resampled samples. Goes from 1 to n_weights.
    """
    @classmethod
    def evaluate(cls, weights):
        weights = _np.asarray(weights, dtype=_np.float32)
        int_n_walkers = _np.floor(weights.shape[0] * weights)
        return int_n_walkers[int_n_walkers >= 1].shape[0]


class ExpWeightEntropy(EffectiveSampleSize):
    """
    Calculates the exponential of the entropy of the normalised weights. Goes from 1 to n_weights.
    """
    @classmethod
    def evaluate(cls, weights):
        weights = _np.asarray(weights, dtype=_np.float32)
        weights /= _np.sum(weights)
        return _np.exp(_entropy(weights))
