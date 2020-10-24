import collections as _collections

import numpy as _np
import pymbar as _pymbar
from scipy.special import logsumexp as _logsumexp


def _memo_factory(always_memo=False):
    def memo(func):
        def memoised_func(self, *args, memo_key=None, **kwargs):
            try:
                return self._memos[func][memo_key]
            except KeyError:
                result = func(self, *args, **kwargs)
                if memo_key is not None or always_memo:
                    self._memos[func][memo_key] = result
                return result

        return memoised_func

    return memo


class MBARResult:
    def __init__(self, u_kn, N_k, bootstrapped_indices=None, initial_f_k=None, **kwargs):
        self._bootstrapped_indices = bootstrapped_indices
        if self._bootstrapped_indices is not None:
            self._bootstrapped_indices = _np.sort(self._bootstrapped_indices)
            N_k_cumsum = _np.zeros(N_k.shape[0] + 1, dtype=_np.int)
            N_k_cumsum[1:] = _np.cumsum(N_k, dtype=_np.int)
            N_k = _np.histogram(self._bootstrapped_indices, N_k_cumsum)[0]
            u_kn = u_kn[N_k != 0, :][:, self._bootstrapped_indices]
            if initial_f_k is not None:
                initial_f_k = initial_f_k[N_k != 0]
            N_k = N_k[N_k != 0]
        self._memos = _collections.defaultdict(dict)
        self._mbar = _pymbar.MBAR(u_kn, N_k, initial_f_k=initial_f_k, **kwargs)
        self._log_weights = -_logsumexp(-u_kn.T + self.getFreeEnergyDifferences(), b=_np.asarray(N_k), axis=1)

    @_memo_factory(True)
    def getFreeEnergyDifferences(self):
        return self._mbar.getFreeEnergyDifferences(compute_uncertainty=False)[0][0]

    @_memo_factory()
    def computePerturbedFreeEnergies(self, u_kn):
        if self._bootstrapped_indices is not None:
            u_kn = u_kn[:, self._bootstrapped_indices]
        return self._mbar.computePerturbedFreeEnergies(u_kn, compute_uncertainty=False)[0][0]

    @_memo_factory()
    def computeExpectations(self, A_n, u_kn=None):
        if u_kn is None:
            u_kn = self._mbar.u_kn
            A_n = A_n[:, self._bootstrapped_indices]
        elif self._bootstrapped_indices is not None:
            u_kn = u_kn[:, self._bootstrapped_indices]
            A_n = A_n[:, self._bootstrapped_indices]
        probabilities = -u_kn + self._log_weights
        probabilities = _np.exp(probabilities - _logsumexp(probabilities))
        return _np.average(A_n, weights=probabilities, axis=1)
