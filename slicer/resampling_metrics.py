import numpy as _np
from scipy.stats import entropy as _entropy


def transform(func):
    def decorator(cls_orig):
        @classmethod
        def defaultValue(cls, n_weights):
            return func(super(cls_orig, cls).defaultValue(n_weights))

        @classmethod
        def evaluate(cls, weights):
            return func(super(cls_orig, cls).evaluate(weights))

        cls_orig.defaultValue = defaultValue
        cls_orig.evaluate = evaluate

        return cls_orig
    return decorator


class EffectiveSampleSize:
    @classmethod
    def defaultValue(cls, n_weights):
        return n_weights / 2

    @classmethod
    def defaultTol(cls, n_weights):
        return 1e-2

    @classmethod
    def evaluate(cls, weights):
        weights = _np.asarray(weights, dtype=_np.float32)
        return _np.sum(weights) ** 2 / _np.sum(weights ** 2)


@transform(_np.log)
class LogEffectiveSampleSize(EffectiveSampleSize):
    pass


class WorstCaseSampleSize(EffectiveSampleSize):
    @classmethod
    def evaluate(cls, weights):
        weights = _np.asarray(weights, dtype=_np.float32)
        return _np.sum(weights) / _np.max(weights)


@transform(_np.log)
class LogWorstCaseSampleSize(WorstCaseSampleSize):
    pass


class WorstCaseSystematicSampleSize(EffectiveSampleSize):
    @classmethod
    def evaluate(cls, weights):
        weights = _np.asarray(weights, dtype=_np.float32)
        int_n_particles = _np.floor(weights.shape[0] * weights)
        return int_n_particles[int_n_particles >= 1].shape[0]


@transform(_np.log)
class LogWorstCaseSystematicSampleSize(WorstCaseSystematicSampleSize):
    pass


class ExpWeightEntropy(EffectiveSampleSize):
    @classmethod
    def evaluate(cls, weights):
        weights = _np.asarray(weights, dtype=_np.float32)
        weights /= _np.sum(weights)
        return _np.exp(_entropy(weights))


@transform(_np.log)
class WeightEntropy(ExpWeightEntropy):
    pass
