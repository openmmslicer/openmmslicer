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


class WorstCaseSampleSize(EffectiveSampleSize):
    @classmethod
    def evaluate(cls, weights):
        weights = _np.asarray(weights, dtype=_np.float32)
        return _np.sum(weights) / _np.max(weights)


class WorstCaseSystematicSampleSize(EffectiveSampleSize):
    @classmethod
    def evaluate(cls, weights):
        weights = _np.asarray(weights, dtype=_np.float32)
        int_n_particles = _np.floor(weights.shape[0] * weights)
        return int_n_particles[int_n_particles >= 1].shape[0]


class ExpWeightEntropy(EffectiveSampleSize):
    @classmethod
    def evaluate(cls, weights):
        weights = _np.asarray(weights, dtype=_np.float32)
        weights /= _np.sum(weights)
        return _np.exp(_entropy(weights))
