import numpy as _np


class EffectiveSampleSize:
    @classmethod
    def defaultValue(cls, n_weights):
        return n_weights / 10

    @classmethod
    def defaultTol(cls, n_weights):
        return 1e-1

    @classmethod
    def evaluate(cls, weights):
        weights = _np.asarray(weights)
        return _np.sum(weights) ** 2 / _np.sum(weights ** 2)


class LogEffectiveSampleSize(EffectiveSampleSize):
    @classmethod
    def defaultValue(cls, n_weights):
        return _np.log(super(LogEffectiveSampleSize, cls).defaultValue(n_weights))

    @classmethod
    def evaluate(cls, weights):
        return _np.log(super(LogEffectiveSampleSize, cls).evaluate(weights))


class WorstCaseSampleSize(EffectiveSampleSize):
    @classmethod
    def evaluate(cls, weights):
        weights = _np.asarray(weights)
        return _np.sum(weights) / _np.max(weights)


class LogWorstCaseSampleSize(WorstCaseSampleSize):
    @classmethod
    def defaultValue(cls, n_weights):
        return _np.log(super(LogWorstCaseSampleSize, cls).defaultValue(n_weights))

    @classmethod
    def evaluate(cls, weights):
        return _np.log(super(LogWorstCaseSampleSize, cls).evaluate(weights))


class ExpWeightEntropy(EffectiveSampleSize):
    @classmethod
    def defaultValue(cls, weights):
        return 0.1

    @classmethod
    def evaluate(cls, weights):
        weights = _np.asarray(weights)
        weights /= _np.sum(weights)
        return _np.exp(-_np.sum(weights * _np.log(weights)))


class WeightEntropy(ExpWeightEntropy):
    @classmethod
    def defaultValue(cls, n_weights):
        return _np.log(super(WeightEntropy, cls).defaultValue(n_weights))

    @classmethod
    def evaluate(cls, weights):
        return _np.log(super(WeightEntropy, cls).evaluate(weights))
