import numpy as _np
from scipy.stats import entropy as _entropy


class EffectiveSampleSize:
    @classmethod
    def defaultValue(cls, n_weights):
        return n_weights / 5

    @classmethod
    def defaultTol(cls, n_weights):
        return 0.1 / n_weights

    @classmethod
    def evaluate(cls, weights):
        weights = _np.asarray(weights, dtype=_np.float32)
        return _np.sum(weights) ** 2 / _np.sum(weights ** 2)


class ExpectedSampleSize(EffectiveSampleSize):
    @classmethod
    def evaluate(cls, weights):
        weights = _np.asarray(weights, dtype=_np.float32)
        weights /= _np.sum(weights)
        n_walkers = weights.shape[0]
        weights *= n_walkers
        weights = weights[weights < 1]
        return n_walkers - (weights.shape[0] - _np.sum(weights))


class WorstCaseSampleSize(EffectiveSampleSize):
    @classmethod
    def evaluate(cls, weights):
        weights = _np.asarray(weights, dtype=_np.float32)
        return _np.sum(weights) / _np.max(weights)


class WorstCaseSystematicSampleSize(EffectiveSampleSize):
    @classmethod
    def evaluate(cls, weights):
        weights = _np.asarray(weights, dtype=_np.float32)
        int_n_walkers = _np.floor(weights.shape[0] * weights)
        return int_n_walkers[int_n_walkers >= 1].shape[0]


class ExpWeightEntropy(EffectiveSampleSize):
    @classmethod
    def evaluate(cls, weights):
        weights = _np.asarray(weights, dtype=_np.float32)
        weights /= _np.sum(weights)
        return _np.exp(_entropy(weights))
