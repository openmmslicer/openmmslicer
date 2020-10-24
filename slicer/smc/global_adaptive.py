import collections as _collections
import functools as _functools
import logging as _logging
import math as _math
import random as _random
import warnings as _warnings

import cma as _cma
import numpy as _np
from scipy.interpolate import interp1d as _interp1d
from scipy.special import logsumexp as _logsumexp
import simtk.unit as _unit

from .cyclic import CyclicSMCSampler as _CyclicSMCSampler
from slicer.alchemy import AbsoluteAlchemicalGaussianSoftcoreFactory as _GaussianSCFactory
from slicer.mbar import MBARResult as _MBARResult

_logger = _logging.getLogger(__name__)


class GlobalAdaptiveCyclicSMCSampler(_CyclicSMCSampler):
    default_alchemical_functions = {
        'lambda_sterics': lambda x: x,
        'lambda_electrostatics': lambda x: max(0., 2. * x - 1.),
        'lambda_torsions': lambda x: max(0., 2. * x - 1.),
    }

    def __init__(self, *args, n_bootstraps=None, **kwargs):
        # TODO: interpolate all energies
        if "alchemical_functions" in kwargs.keys():
            _warnings.warn("Custom alchemical functions are not supported. Switching to the defaults...")
        if "fe_estimator" in kwargs.keys():
            _warnings.warn("Custom free energy estimators are not supported. Switching to MBAR...")
            kwargs.pop("fe_estimator")
        kwargs["alchemical_functions"] = GlobalAdaptiveCyclicSMCSampler.default_alchemical_functions
        super().__init__(*args, **kwargs)
        self.n_bootstraps = n_bootstraps
        # TODO: clean up memos
        self._next_optimisation = None
        self._next_fe_calculation = None
        self._last_memo_update = 0
        self._fe_memo = None
        self._interpol_memo = _collections.defaultdict(lambda: {
            0.: [],
            0.5: [],
            1.: [],
            "w": [],
            "iter": 0,
        })
        self._total_interpol_memo = {}
        self._protocol_memo = {}

    @property
    def current_lambdas(self):
        # TODO: make prettier
        try:
            return self._current_lambdas
        except AttributeError:
            return super().current_lambdas

    @current_lambdas.setter
    def current_lambdas(self, val):
        self._current_lambdas = val

    @property
    def unique_lambdas(self):
        return sorted(set(self.lambda_history))

    def _update_interpol_memos(self):
        if len(self.lambda_history) <= self._last_memo_update:
            return

        # update the disordered cache
        relevant_lambdas = set(self.lambda_history[self._last_memo_update:])
        for lambda_ in relevant_lambdas:
            min_iter = self._interpol_memo[lambda_]["iter"]
            walkers = [w for w in self._all_walkers
                       if w.lambda_ is not None and _math.isclose(w.lambda_, lambda_) and w.iteration >= min_iter]
            walkers.sort(key=lambda x: x.iteration)
            iterations = [w.iteration for w in walkers]
            counters = _collections.Counter(iterations)
            weights = [1 / counters[i] for i in iterations]
            for k in [0., 0.5, 1.]:
                self._interpol_memo[lambda_][k] += self.calculateStateEnergies(k, walkers=walkers).tolist()
            self._interpol_memo[lambda_]["w"] += weights
            self._interpol_memo[lambda_]["iter"] = max(w.iteration for w in walkers) + 1
        self._last_memo_update = len(self.lambda_history)

        # update the ordered cache
        lambdas = sorted(self._interpol_memo.keys())
        self._total_interpol_memo = {}
        for k in [0., 0.5, 1., "w"]:
            self._total_interpol_memo[k] = _np.asarray(sum([self._interpol_memo[x][k] for x in lambdas], []))
        self._total_interpol_memo["n"] = _np.asarray([len(self._interpol_memo[x][0.]) for x in lambdas])

    def calculateTotalStateEnergies(self, lambda_):
        self._update_interpol_memos()
        if lambda_ in [0., 0.5, 1.]:
            E = self._total_interpol_memo[lambda_]
        elif 0 < lambda_ < 0.5:
            E = self._total_interpol_memo[0.] * (1 - 2 * lambda_) + self._total_interpol_memo[0.5] * (2 * lambda_)
        elif 0.5 < lambda_ < 1:
            E = self._total_interpol_memo[0.5] * (2 - 2 * lambda_) + self._total_interpol_memo[1.] * (2 * lambda_ - 1)
        else:
            raise ValueError("Lambda value {} not in the range [0, 1]".format(lambda_))
        return E

    def energyMatrix(self, lambdas=None):
        # TODO: make calling the memos more robust
        self._update_interpol_memos()
        if lambdas is None:
            lambdas = self.unique_lambdas
        M = _np.zeros((len(lambdas), self._total_interpol_memo[0.].shape[0]))
        for i, lambda_ in enumerate(lambdas):
            M[i, :] = self.calculateTotalStateEnergies(lambda_)
        return M

    def MBAR(self, recalculate=False, **kwargs):
        calculate = False
        if recalculate or self._next_fe_calculation is None or self._next_fe_calculation <= len(self.lambda_history):
            calculate = True
            self._next_fe_calculation = len(self.lambda_history) + len(self.current_lambdas) - 1

        if calculate:
            lambdas = _np.asarray(self.unique_lambdas)
            u_kn = self.energyMatrix()
            N_k = self._total_interpol_memo["n"]
            weights = self._total_interpol_memo["w"]
            N_resample = _np.round(_np.sum(weights)).astype(_np.int)
            weights /= _np.sum(weights)

            if self._fe_memo is not None:
                lambdas_old, mbars_old = self._fe_memo
                interp = _interp1d(lambdas_old, mbars_old[0].computePerturbedFreeEnergies(memo_key=tuple(lambdas_old)))
            else:
                interp = _interp1d([0., 1.], [0., 0.])
            kwargs["initial_f_k"] = interp(lambdas)

            if self.n_bootstraps is not None:
                ids = _np.random.choice(_np.arange(u_kn.shape[1]), size=(self.n_bootstraps, N_resample), p=weights)
            else:
                ids = [_np.arange(u_kn.shape[1])]
            mbars = [_MBARResult(u_kn, N_k, bootstrapped_indices=x, **kwargs) for x in ids]
            u_kn_pert = self.energyMatrix(self.current_lambdas)
            for mbar in mbars:
                mbar.computePerturbedFreeEnergies(u_kn_pert, memo_key=tuple(self.current_lambdas))
            self._fe_memo = self.current_lambdas, mbars

        return self._fe_memo

    @ property
    def fe_estimator(self):
        def func(lambda1, lambda0, **kwargs):
            lambdas, mbars = self.MBAR(**kwargs)
            if lambda1 not in self.current_lambdas or lambda0 not in self.current_lambdas:
                memo_key = (lambda0, lambda1)
            else:
                memo_key = tuple(self.current_lambdas)
            fes = []
            for mbar in mbars:
                try:
                    result = mbar.computePerturbedFreeEnergies(memo_key=memo_key)
                except TypeError:
                    u_kn_part = self.energyMatrix(memo_key)
                    result = mbar.computePerturbedFreeEnergies(u_kn_part, memo_key=memo_key)
                fes += [result[memo_key.index(lambda1)] - result[memo_key.index(lambda0)]]
            return _np.asarray(fes)
        return func

    @fe_estimator.setter
    def fe_estimator(self, val):
        pass

    def expectedTransitionMatrix(self, lambdas, mbar):
        if len(lambdas) < 2:
            raise ValueError("Need a lambda array of at least two lambda values")
        lambdas = _np.asarray(lambdas)
        lambdas[lambdas < 0] = 0.
        lambdas[lambdas > 1] = 1.

        u_kn = self.energyMatrix(lambdas)
        fes = mbar.computePerturbedFreeEnergies(u_kn, memo_key=tuple(lambdas))
        log_probabilities = -u_kn + fes.reshape(-1, 1)

        size = 2 * len(lambdas) - 2
        T = _np.zeros((size, size))
        for i, lambda_ in enumerate(lambdas):
            u_kn_i = u_kn[[i], :]
            if i != len(lambdas) - 1:
                T_fwd_cache = _np.exp(_np.minimum(0., log_probabilities[[i + 1]] - log_probabilities[[i]]))
                T[2 * i, min(2 * i + 2, size - 1)] = mbar.computeExpectations(T_fwd_cache, u_kn_i)
            if i != 0:
                T_bwd_cache = _np.exp(_np.minimum(0., log_probabilities[[i - 1]] - log_probabilities[[i]]))
                T[2 * i - 1, max(2 * i - 3, 0)] = mbar.computeExpectations(T_bwd_cache, u_kn_i)
            if i != 0 and i != len(lambdas) - 1:
                T[2 * i, 2 * i - 1] = mbar.computeExpectations(_np.maximum(0., T_bwd_cache - T_fwd_cache), u_kn_i)
                T[2 * i - 1, 2 * i] = mbar.computeExpectations(_np.maximum(0., T_fwd_cache - T_bwd_cache), u_kn_i)
        for i in range(size):
            T[i, i] = 1. - _np.sum(T[i, :])

        return T

    def expectedRoundTripTime(self, lambdas, *args, **kwargs):
        lambdas = _np.asarray(lambdas)
        lambdas[lambdas < 0] = 0.
        lambdas[lambdas > 1] = 1.
        T = self.expectedTransitionMatrix(lambdas, *args, **kwargs)
        T_fwd = T[1:, 1:]
        T_bwd = T[:-1, :-1]
        costs = [self.decorrelationSteps(lambda_) for lambda_ in lambdas]
        costs = _np.asarray([costs[0]] + [x for x in costs[1:-1] for _ in range(2)] + [costs[1]])
        identity = _np.identity(T_fwd.shape[0])
        b = T @ costs
        try:
            tau_fwd = _np.linalg.solve(identity - T_fwd, b[1:])[-1]
            tau_bwd = _np.linalg.solve(identity - T_bwd, b[:-1])[0]
            tau_total = tau_bwd + tau_fwd
        except _np.linalg.LinAlgError:
            tau_total = _np.inf
        if tau_total < 0:
            _logger.warning(f"The expected round trip time for the protocol {lambdas} was {tau_total}")
            tau_total = _np.inf
        return tau_total

    def _closest_protocol(self, N, fixed_lambdas=None):
        if fixed_lambdas is None:
            fixed_lambdas = []
        fixed_lambdas = sorted(set(fixed_lambdas) | {0., 1., self.lambda_})
        if N in self._protocol_memo.keys():
            protocol = self._protocol_memo[N]
        elif any(k for k in self._protocol_memo.keys() if k > N):
            protocol = self._protocol_memo[sorted(k for k in self._protocol_memo.keys() if k > N)[0]]
        elif any(k for k in self._protocol_memo.keys() if k < N):
            protocol = self._protocol_memo[sorted(k for k in self._protocol_memo.keys() if k < N)[-1]]
        else:
            protocol = self.current_lambdas
        protocol = _interp1d(_np.linspace(0., 1., num=len(protocol)), protocol)(_np.linspace(0., 1., num=N)).tolist()
        for lambda_ in fixed_lambdas:
            protocol.remove(min(protocol, key=lambda x: abs(x - lambda_)))
        return sorted(protocol + fixed_lambdas)

    def _augment_fixed_lambdas(self, fixed_lambdas=None, start=0., end=1.):
        if fixed_lambdas is None:
            fixed_lambdas = []
        fixed_lambdas = set(fixed_lambdas) | {x for x in self.current_lambdas if not start < x < end} | {self.lambda_}
        return sorted(fixed_lambdas)

    def _continuous_optimise_protocol(self, n_opt, fixed_lambdas=None, start=0., end=1., tol=1e-3, **kwargs):
        fixed_lambdas = self._augment_fixed_lambdas(fixed_lambdas=fixed_lambdas, start=start, end=end)
        internal_fixed_lambdas = [x for x in fixed_lambdas if start <= x <= end]

        def f(x):
            x = _np.asarray(x)
            x[x < x_start] = x_start
            x[x > x_end] = x_end
            x = _np.sort(interp(x).tolist() + internal_fixed_lambdas)
            return _np.average([self.expectedRoundTripTime(x, m) for m in self.MBAR()[-1]])

        N = max(n_opt, 0)
        N_total = N + len(fixed_lambdas)
        N_internal = N + len(internal_fixed_lambdas)

        if N == 0:
            protocol = fixed_lambdas
            fun = _np.average([self.expectedRoundTripTime(_np.asarray(internal_fixed_lambdas), m) for m in self.MBAR()[-1]])
        else:
            # interpolate from the closest protocol
            y_interp = _np.asarray(self._closest_protocol(N_total, fixed_lambdas=fixed_lambdas))
            x_interp = _np.linspace(0., 1., num=len(y_interp))
            interp = _interp1d(x_interp, y_interp)

            # define some initial parameters for the minimiser
            x_start, x_end = *x_interp[y_interp == start], *x_interp[y_interp == end]
            x0 = x_interp[~_np.isin(y_interp, fixed_lambdas, assume_unique=True)]
            x0[x0 < x_start] = x_start + (x_end - x_start) / N
            x0[x0 > x_end] = x_end - (x_end - x_start) / N
            kwargs = {**dict(verb_log=False, verb_log_expensive=False, bounds=([x_start] * N, [x_end] * N),
                             maxfevals=10 * N, tolfun=tol), **kwargs}
            sigma0 = 0.25 * (x_end - x_start) / (N_internal - 2)

            # minimise
            if x0.shape[0] == 1:
                val = _cma.fmin(lambda x: f(x[:1]), _np.asarray(list(x0) * 2), sigma0=sigma0, options=kwargs)
                protocol = [val[5][0]]
            else:
                val = _cma.fmin(f, _np.asarray(x0), sigma0=sigma0, options=kwargs)
                protocol = list(val[5])
            fun = f(protocol)
            protocol = sorted(interp(protocol).tolist() + fixed_lambdas)
        self._protocol_memo[N_total] = protocol
        return protocol, fun

    def _discrete_optimise_protocol(self, fixed_lambdas=None, start=0., end=1., tol=1e-3, **kwargs):
        fixed_lambdas = self._augment_fixed_lambdas(fixed_lambdas=fixed_lambdas, start=start, end=end)

        # minimise the lambda values given an integer number of them
        @_functools.lru_cache(maxsize=None)
        def optfunc(n_opt):
            return self._continuous_optimise_protocol(n_opt, fixed_lambdas=fixed_lambdas, start=start, end=end,
                                                      tol=tol, **kwargs)

        # minimise the number of lambda windows
        N_adapt = len([x for x in self.current_lambdas if x not in fixed_lambdas])
        while True:
            protocol_list = [(N_adapt + i, *optfunc(N_adapt + i)) for i in [0, 1, -1]]
            min_protocol = min(protocol_list, key=lambda x: x[-1])
            if min_protocol[0] == N_adapt:
                protocol_curr, fun_curr = min_protocol[1:]
                if _np.isinf(fun_curr):
                    N_adapt += 1
                else:
                    break
            else:
                N_adapt = min_protocol[0]

        return protocol_curr, fun_curr

    def optimiseProtocol(self, start=None, end=None, **kwargs):
        if self._next_optimisation is None:
            self.current_lambdas = self.lambda_history[:self.lambda_history.index(1) + 1]
            self._next_optimisation = len(self.lambda_history) + 5 * (2 * len(self.current_lambdas) - 2)
            self._prev_optimisation_index = len(self.lambda_history)
        elif self._next_optimisation <= len(self.lambda_history):
            start_suggested, end_suggested = self.effective_lambda_bounds
            start = start_suggested if start is None else start
            end = end_suggested if end is None else end

            protocol, fun = self._discrete_optimise_protocol(start=start, end=end, **kwargs)
            self.current_lambdas = protocol
            t = fun * self.integrator.getStepSize().in_units_of(_unit.nanosecond)
            _logger.info(f"The optimal protocol after adaptation between lambda = {start} and lambda = {end} with an "
                         f"expected round trip time of {t} is: {self.current_lambdas}")

            self._next_optimisation = len(self.lambda_history) + 5 * (2 * len(self.current_lambdas) - 2) * (1 + self.cycles)
            self._prev_optimisation_index = len(self.lambda_history)

    def reweight(self, *args, **kwargs):
        if self.adaptive_mode:
            return super().reweight(*args, **kwargs)

        # get the protocol schedule from the adaptive step
        idx = next(i for i, x in enumerate(self.current_lambdas) if _math.isclose(x, self.lambda_))
        sign = -1 if self.lambda_ != 0 and (self.target_lambda > self.lambda_ or self.lambda_ == 1) else 1
        prev_lambda = self.current_lambdas[(idx + sign)]
        sign = 1 if self.lambda_ != 1 and (self.target_lambda > self.lambda_ or self.lambda_ == 0) else -1
        next_lambda = self.current_lambdas[(idx + sign)]

        # get acceptance criterion
        fe_fwd = self.fe_estimator(next_lambda, self.lambda_)
        deltaEs_fwd = self.calculateDeltaEs(next_lambda, self.lambda_)
        deltaEs_fwd = -_logsumexp(-deltaEs_fwd) + _np.log(deltaEs_fwd.shape[0])
        target_lambda = 1 if self.lambda_history[::-1].index(1) > self.lambda_history[::-1].index(0) else 0
        fe_fwd = _np.max(fe_fwd) if target_lambda >= next_lambda >= self.lambda_ else _np.min(fe_fwd)
        samples_fwd = _np.sum(_np.meshgrid(fe_fwd, -deltaEs_fwd), axis=0)
        acc_fwd = _np.average(_np.exp(_np.minimum(samples_fwd, 0.)))
        _logger.debug("Forward probability: {}".format(acc_fwd))

        # accept or reject move and/or swap direction
        randnum = _random.random()
        if acc_fwd != 1. and randnum >= acc_fwd:
            if not _math.isclose(next_lambda, prev_lambda):
                fe_bwd = self.fe_estimator(prev_lambda, self.lambda_)
                target_lambda = 1 if self.lambda_history[::-1].index(1) > self.lambda_history[::-1].index(0) else 0
                fe_bwd = _np.max(fe_bwd) if target_lambda >= prev_lambda >= self.lambda_ else _np.min(fe_bwd)
                deltaEs_bwd = self.calculateDeltaEs(prev_lambda, self.lambda_)
                deltaEs_bwd = -_logsumexp(-deltaEs_bwd) + _np.log(deltaEs_bwd.shape[0])
                samples_bwd = _np.sum(_np.meshgrid(fe_bwd, -deltaEs_bwd), axis=0)
                acc_bwd = max(0., _np.average(_np.exp(_np.minimum(samples_bwd, 0.))) - acc_fwd)
                _logger.debug("Backward probability: {}".format(acc_bwd))

                if acc_bwd != 0. and randnum - acc_fwd < acc_bwd:
                    self.target_lambda = int(not self.target_lambda)
            # trigger walker update
            self.lambda_ = self.lambda_
        else:
            self.lambda_ = next_lambda

        return self.lambda_

    @property
    def effective_lambda_bounds(self):
        lambdas_recent = self.lambda_history[self._prev_optimisation_index:]
        min_lambda, max_lambda = min(lambdas_recent), max(lambdas_recent)
        min_lambda_idx, max_lambda_idx = self.current_lambdas.index(min_lambda), self.current_lambdas.index(max_lambda)
        left = self.current_lambdas[max(0, min_lambda_idx - 1)]
        right = self.current_lambdas[min(max_lambda_idx + 1, len(self.current_lambdas) - 1)]
        return left, right

    def _runPostAdaptiveIteration(self, optimise_kwargs=None, **kwargs):
        if optimise_kwargs is None:
            optimise_kwargs = {}
        self.optimiseProtocol(**optimise_kwargs)
        super()._runPostAdaptiveIteration(**kwargs)

    @staticmethod
    def generateAlchSystem(*args, **kwargs):
        if "alchemical_factory" in kwargs.keys():
            if not isinstance(kwargs["alchemical_factory"], _GaussianSCFactory):
                _warnings.warn("Alchemical factory not supported. Switching to "
                               "AbsoluteAlchemicalGaussianSoftcoreFactory...")
        kwargs["alchemical_factory"] = _GaussianSCFactory
        return super(GlobalAdaptiveCyclicSMCSampler, GlobalAdaptiveCyclicSMCSampler).generateAlchSystem(*args, **kwargs)
