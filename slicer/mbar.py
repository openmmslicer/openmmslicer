import collections as _collections
import logging as _logging
import sys as _sys

import numpy as _np
import pymbar as _pymbar
from scipy.special import logsumexp as _logsumexp

_logger = _logging.getLogger(__name__)


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
    def __init__(self, u_kn, N_k, train_indices=None, test_indices=None, u_kn_test=None, N_k_test=None,
                 initial_f_k=None, **kwargs):
        self._u_kn = u_kn
        N_k_test = N_k if N_k_test is None else N_k_test
        u_kn_test = self._u_kn if u_kn_test is None else u_kn_test
        self._train_indices = train_indices
        if self._train_indices is not None:
            self._train_indices = _np.sort(self._train_indices)
            N_k_train = self._getSampleSize(N_k, self._train_indices)
            u_kn = u_kn[N_k_train != 0, :][:, self._train_indices]
            if initial_f_k is not None:
                initial_f_k = initial_f_k[N_k_train != 0]
            N_k_train = N_k_train[N_k_train != 0]
        else:
            self._train_indices = _np.arange(u_kn.shape[1])
            N_k_train = N_k
        self._test_indices = self._train_indices if test_indices is None else test_indices
        self._memos = _collections.defaultdict(dict)

        # TODO: deal with the rare case of an SVD error
        stdout, stderr = _sys.stdout.write, _sys.stderr.write
        _sys.stdout.write, _sys.stderr.write = _logger.debug, _logger.error
        self._mbar = _pymbar.MBAR(u_kn, N_k_train, initial_f_k=initial_f_k, **kwargs)
        _sys.stdout.write, _sys.stderr.write = stdout, stderr

        N_k_test = self._getSampleSize(N_k_test, self._test_indices)
        u_kn = u_kn_test[:, self._train_indices]
        fes = self._mbar.computePerturbedFreeEnergies(u_kn, compute_uncertainty=False)[0][0]
        u_kn_test = u_kn_test[:, self._test_indices]
        self._log_weights = -_logsumexp(-u_kn_test.T + fes, b=_np.asarray(N_k_test), axis=1)

    @_memo_factory(True)
    def getFreeEnergyDifferences(self):
        if self._test_indices is None:
            return self._mbar.getFreeEnergyDifferences(compute_uncertainty=False)[0][0]
        else:
            return self.computePerturbedFreeEnergies(self._u_kn)

    @_memo_factory()
    def computePerturbedFreeEnergies(self, u_kn):
        # TODO: throw errors if shape mismatch between cached energies and passed energies
        assert u_kn.shape[1] == self._u_kn.shape[1]
        u_kn = u_kn[:, self._test_indices]
        _log_weights = self._log_weights
        fes = -_logsumexp(-u_kn + _log_weights, axis=1)
        fes -= fes[0]
        return fes

    @_memo_factory()
    def computeExpectations(self, A_n, u_kn=None):
        if u_kn is None:
            u_kn = self._u_kn
        # TODO: throw errors if shape mismatch between cached energies and passed energies
        assert u_kn.shape[1] == A_n.shape[1] == self._u_kn.shape[1]
        if self._test_indices is not None:
            u_kn = u_kn[:, self._test_indices]
            A_n = A_n[:, self._test_indices]
        log_weights = self._log_weights

        probabilities = -u_kn + log_weights
        probabilities = _np.exp(probabilities - _logsumexp(probabilities))
        return _np.average(A_n, weights=probabilities, axis=1)

    @staticmethod
    def _getSampleSize(N_k, indices):
        N_k_cumsum = _np.zeros(N_k.shape[0] + 1, dtype=_np.int)
        N_k_cumsum[1:] = _np.cumsum(N_k, dtype=_np.int)
        return _np.histogram(indices, N_k_cumsum)[0]


class IterativeMBARResult:
    def __init__(self, u_kn, N_k, initial_f_k=None, **kwargs):
        self._mbar = _pymbar.MBAR(u_kn, N_k, initial_f_k=initial_f_k, **kwargs)
        fes = self._mbar.computePerturbedFreeEnergies(u_kn, compute_uncertainty=False)[0][0]
        self._log_weights = (-_logsumexp(-u_kn.T + fes, b=_np.asarray(N_k), axis=1) + _np.log(_np.sum(N_k))).tolist()
        self._fe_memo = {}

    def addSamples(self, u_kn, N_k, f_k, lambdas=None):
        if not u_kn.size:
            return
        f_k -= f_k[0]
        log_weights = -_logsumexp(-u_kn.T + f_k, b=_np.asarray(N_k), axis=1) + _np.log(_np.sum(N_k))
        self._log_weights = _np.concatenate([self._log_weights, log_weights])
        if lambdas is None:
            self._fe_memo = {}
        else:
            self._fe_memo = {k: v for k, v in self._fe_memo.items() if k in lambdas}
            for lambda_, (n_samples, fe) in self._fe_memo.items():
                idx = _np.where(lambdas == lambda_)
                N = [n_samples] + [1] * u_kn.shape[1]
                fe = -_logsumexp([-fe] + (log_weights - u_kn[idx]).tolist() , b=N) + _np.log(_np.sum(N))
                self._fe_memo[lambda_] = (n_samples, fe)

    def getFreeEnergyDifferences(self, lambdas=None):
        if lambdas is None:
            if not self._fe_memo:
                fes = self._mbar.getFreeEnergyDifferences(compute_uncertainty=False)[0][0]
            else:
                fes = _np.asarray([v[-1] for k, v in self._fe_memo.items()])
        else:
            fes = _np.asarray([self._fe_memo[lambda_] for lambda_ in lambdas])
        fes -= fes[0]
        return fes

    def computePerturbedFreeEnergies(self, u_kn, lambdas=None):
        fes = -_logsumexp(-u_kn + self._log_weights, axis=1)
        fes -= fes[0]
        if False and lambdas is not None:
            self._fe_memo.update({lambda_: (len(self._log_weights), fes[i]) for i, lambda_ in enumerate(lambdas)})
        return fes

    def computeExpectations(self, A_n, u_kn):
        probabilities = -u_kn + self._log_weights
        probabilities = _np.exp(probabilities - _logsumexp(probabilities))
        return _np.average(A_n, weights=probabilities, axis=1)