import abc as _abc
import math as _math

import numpy as _np
import pymbar as _pymbar
import scipy.optimize as _optimize


class AbstractFEEstimator(_abc.ABC):
    def __init__(self, ensemble):
        self.ensemble = ensemble

    @property
    def ensemble(self):
        return self._ensemble

    @ensemble.setter
    def ensemble(self, val):
        self._ensemble = val
        self._deltaE_memo = {}
        self._FE_memo = {}

    @staticmethod
    def _isclose(x, y):
        return all(_math.isclose(*items) for items in zip(x, y))

    def _getDeltaE(self, lambda1, lambda0):
        # TODO: correctly calculate FE's from weighted walkers

        # preliminary checks
        key = (lambda1, lambda0)
        if not any(self._isclose(key, x) for x in self._deltaE_memo.keys()):
            self._deltaE_memo[key] = [[], 0]
        else:
            key = next(x for x in self._deltaE_memo.keys() if self._isclose(key, x))

        # only get the uncached walkers
        min_iter = self._deltaE_memo[key][-1]
        walkers = [w for w in self.ensemble._all_walkers
                   if w.lambda_ is not None and _math.isclose(w.lambda_, key[-1]) and w.iteration >= min_iter]

        # update the cache
        self._deltaE_memo[key][0] += list(self.ensemble.calculateDeltaEs(*key, walkers))
        works = _np.asarray(self._deltaE_memo[key][0])
        if len(walkers):
            self._deltaE_memo[key][-1] = max(x.iteration for x in walkers) + 1

        return works

    def _getFE(self, lambda1, lambda0):
        key = (lambda1, lambda0)
        if not any(self._isclose(key, x) for x in self._FE_memo.keys()):
            return None
        else:
            return self._FE_memo[next(x for x in self._FE_memo.keys() if self._isclose(key, x))]

    def _updateFE(self, lambda1, lambda0, fe):
        for i in [-1, 1]:
            key = (lambda1, lambda0)[::i]
            if any(self._isclose(key, x) for x in self._FE_memo.keys()):
                key = next(x for x in self._FE_memo.keys() if self._isclose(key, x))
            self._FE_memo[key] = i * fe

    @_abc.abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class BAR(AbstractFEEstimator):
    def __call__(self, lambda1, lambda0, **kwargs):
        # TODO: possibly use FastMBAR

        DeltaF = self._getFE(lambda1, lambda0)
        if DeltaF is not None:
            kwargs["DeltaF"] = DeltaF
        kwargs["method"] = "self-consistent-iteration"
        kwargs["compute_uncertainty"] = False
        fe = _pymbar.bar.BAR(self._getDeltaE(lambda1, lambda0), self._getDeltaE(lambda0, lambda1), **kwargs)
        self._updateFE(lambda1, lambda0, fe)

        return fe


class EqualMetropolis(AbstractFEEstimator):
    def __call__(self, lambda1, lambda0, **kwargs):
        w_F, w_R = self._getDeltaE(lambda1, lambda0), self._getDeltaE(lambda0, lambda1)

        a = _np.max(w_F)
        b = - _np.max(w_R)

        def optfunc(x):
            metr_fwd = _np.average(_np.exp(_np.minimum(-w_F + x, 0.)))
            metr_bwd = _np.average(_np.exp(_np.minimum(-w_R - x, 0.)))
            return metr_fwd - metr_bwd

        fe = _optimize.bisect(optfunc, a, b, **kwargs)
        self._updateFE(lambda1, lambda0, fe)

        return fe
