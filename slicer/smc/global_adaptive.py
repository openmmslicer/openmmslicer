import collections as _collections
import functools as _functools
import logging as _logging
import math as _math
import sys as _sys
import threading as _threading
import warnings as _warnings

import cma as _cma
import numpy as _np
from scipy.interpolate import interp1d as _interp1d
import simtk.unit as _unit

from .cyclic import CyclicSMCSampler as _CyclicSMCSampler
from slicer.alchemy import AbsoluteAlchemicalGaussianSoftcoreFactory as _GaussianSCFactory
from slicer.interpolate import BatchLinearInterp as _BatchLinearInterp
from slicer.mbar import MBARResult as _MBARResult

_logger = _logging.getLogger(__name__)


class GlobalAdaptiveCyclicSMCSampler(_CyclicSMCSampler):
    default_alchemical_functions = {
        'lambda_sterics': lambda x: x,
        'lambda_electrostatics': lambda x: max(0., 2. * x - 1.),
        'lambda_torsions': lambda x: max(0., 2. * x - 1.),
    }

    default_pymbar_kwargs = dict(solver_protocol=(dict(method=None, tol=1e-8, options=dict(maximum_iterations=100)),))

    def __init__(self, *args, n_bootstraps_fe=1, n_bootstraps_opt=1, decorrelate_fe=True, decorrelate_opt=True,
                 n_decorr_fe=10, n_decorr_opt=10, min_update_fe=1, min_update_opt=100, freq_update_fe=0.01, freq_opt=0.1,
                 significant_lambda_figures=2, pymbar_kwargs=None, parallel=False, **kwargs):
        if "alchemical_functions" in kwargs.keys() and kwargs["alchemical_functions"] is not None:
            _warnings.warn("Custom alchemical functions are not supported. Switching to the defaults...")
        kwargs["alchemical_functions"] = GlobalAdaptiveCyclicSMCSampler.default_alchemical_functions
        if "fe_estimator" in kwargs.keys():
            _warnings.warn("Custom free energy estimators are not supported. Switching to MBAR...")
            kwargs.pop("fe_estimator")
        super().__init__(*args, **kwargs)
        if pymbar_kwargs is None:
            pymbar_kwargs = {}
        self.pymbar_kwargs = {**self.default_pymbar_kwargs, **pymbar_kwargs}
        self.parallel = parallel
        self.n_bootstraps_fe = n_bootstraps_fe
        self.n_bootstraps_opt = n_bootstraps_opt
        self.decorrelate_fe = decorrelate_fe
        self.decorrelate_opt = decorrelate_opt
        self.n_decorr_fe = n_decorr_fe
        self.n_decorr_opt = n_decorr_opt
        self.min_update_fe = min_update_fe
        self.min_update_opt = min_update_opt
        self.freq_update_fe = freq_update_fe
        self.freq_opt = freq_opt
        self.significant_lambda_figures = significant_lambda_figures
        # TODO: clean up memos
        self._next_optimisation = None
        self._next_tau_calculation = None
        self._last_interpol_memo_update = 0
        self._last_total_interpol_memo_update = 0
        self._fe_memo = {}
        self._decorr_memo = None
        self._interpol_memo = _collections.defaultdict(lambda: {
            "interp": [],
            "x": [],
            "y": [],
            "w": [],
            "i": [],
            "iter": 0,
        })
        self._total_interpol_memo = {}
        self._protocol_memo = {}
        self._optimisation_memo = [0]

    @property
    def current_lambdas(self):
        try:
            return self._current_lambdas
        except AttributeError:
            return super().current_lambdas

    @current_lambdas.setter
    def current_lambdas(self, val):
        self._current_lambdas = sorted({x for x in set(val) if 0 <= x <= 1} | {0., 1., self.lambda_})

    @property
    def effective_lambda_bounds(self):
        lambdas_recent = self.lambda_history[self._prev_optimisation_index:]
        min_lambda, max_lambda = min(lambdas_recent), max(lambdas_recent)
        min_lambda_idx, max_lambda_idx = self.current_lambdas.index(min_lambda), self.current_lambdas.index(max_lambda)
        left = self.current_lambdas[max(0, min_lambda_idx - 1)]
        right = self.current_lambdas[min(max_lambda_idx + 1, len(self.current_lambdas) - 1)]
        return left, right

    @_CyclicSMCSampler.lambda_.setter
    def lambda_(self, val):
        if val is not None:
            val = round(val, self.significant_lambda_figures)
            _CyclicSMCSampler.lambda_.fset(self, val)

    @property
    def unique_lambdas(self):
        return sorted(set(self.lambda_history))

    def _time_to_mbar_indices(self, indices):
        weights = self._total_interpol_memo["w"]
        argsort = _np.argsort(self._total_interpol_memo["i"])
        cum_weights = _np.concatenate(([0.], _np.cumsum(weights[argsort])))
        diff = _np.abs(cum_weights - _np.round(cum_weights, 0))
        indices_int = _np.where(diff < 1e-8)[0]
        indices_new = [sum([list(range(indices_int[y], indices_int[y + 1])) for y in x], []) for x in indices]
        indices_new = [argsort[x] for x in indices_new]
        return indices_new

    def _update_interpol_memos(self):
        if len(self.lambda_history) <= self._last_interpol_memo_update:
            return

        # update the disordered cache
        relevant_lambdas = set(self.lambda_history[self._last_interpol_memo_update:])
        for lambda_ in relevant_lambdas:
            min_iter = self._interpol_memo[lambda_]["iter"]
            walkers = [(i, w) for i, w in enumerate(self._all_walkers) if w.lambda_ is not None
                       and _math.isclose(w.lambda_, lambda_) and w.iteration >= min_iter and w.transform is None]
            indices, walkers = [list(x) for x in zip(*walkers)]
            walkers.sort(key=lambda x: x.iteration)
            iterations = [w.iteration for w in walkers]
            counters = _collections.Counter(iterations)
            weights = [1 / counters[i] for i in iterations]
            # TODO: obtain interp_lambdas correctly
            interp_lambdas = sorted({x for x in self.current_lambdas if x < 0.5} | {0.5, 1.})
            energies = self.calculateStateEnergies(interp_lambdas, walkers=walkers)
            self._interpol_memo[lambda_]["x"] += [interp_lambdas] * len(walkers)
            self._interpol_memo[lambda_]["y"] += energies.T.tolist()
            self._interpol_memo[lambda_]["w"] += weights
            self._interpol_memo[lambda_]["i"] += indices
            self._interpol_memo[lambda_]["iter"] = max(w.iteration for w in walkers) + 1
        self._last_interpol_memo_update = len(self.lambda_history)

    def _update_total_interpol_memos(self):
        # update the ordered cache
        if self._last_interpol_memo_update == self._last_total_interpol_memo_update:
            return
        lambdas = sorted(self._interpol_memo.keys())
        self._total_interpol_memo = {}
        for k in ["x", "y", "w", "i"]:
            self._total_interpol_memo[k] = _np.asarray(sum([self._interpol_memo[x][k] for x in lambdas], []))
        self._total_interpol_memo["interp"] = _BatchLinearInterp(*[self._total_interpol_memo[k] for k in ["x", "y"]])
        sorted_indices = _np.argsort(self._total_interpol_memo["i"])
        self._total_interpol_memo["i"][sorted_indices] = _np.arange(self._total_interpol_memo["i"].shape[0])
        self._total_interpol_memo["n"] = _np.asarray([len(self._interpol_memo[x]["i"]) for x in lambdas], dtype=_np.int)
        self._last_total_interpol_memo_update = self._last_interpol_memo_update

    def calculateStateEnergies(self, lambda_=None, walkers=None, **kwargs):
        """
        Calculates the reduced potential energies of all states for a given lambda value.

        Parameters
        ----------
        lambda_ : float
            The desired lambda value.
        walkers : [int] or [openmm.State] or None
            Which walkers need to be used. If None, self.walkers are used. Otherwise, these could be in any
            format supported by setState().
        kwargs
            Keyword arguments to be passed to setState().
        """
        if walkers is None:
            walkers = self.walkers
        if lambda_ is None:
            lambdas = _np.asarray([walker.lambda_ for walker in walkers])
        else:
            lambdas = _np.full(len(walkers), lambda_)
        unique_lambdas = _np.unique(lambdas)
        energies = _np.zeros(len(walkers))

        for lambda_ in unique_lambdas:
            indices = _np.where(lambdas == lambda_)[0]
            kwargs["walkers"] = [walkers[i] for i in indices]
            if lambda_ in [0, 0.5, 1]:
                current_energies = super().calculateStateEnergies(lambda_, **kwargs)
            elif 0 < lambda_ < 0.5:
                current_energies_0 = super().calculateStateEnergies(0, **kwargs)
                current_energies_half = super().calculateStateEnergies(0.5, **kwargs)
                current_energies = current_energies_0 * (1 - 2 * lambda_) + current_energies_half * (2 * lambda_)
            elif 0.5 < lambda_ < 1:
                current_energies_half = super().calculateStateEnergies(0.5, **kwargs)
                current_energies_1 = super().calculateStateEnergies(1, **kwargs)
                current_energies = current_energies_half * (2 - 2 * lambda_) + current_energies_1 * (2 * lambda_ - 1)
            else:
                raise ValueError("Lambda value {} not in the range [0, 1]".format(lambda_))
            energies[indices] = current_energies

        return energies

    def calculateTotalStateEnergies(self, lambdas):
        self._update_total_interpol_memos()
        E = self._total_interpol_memo['interp'](_np.asarray(lambdas))
        return E

    def energyMatrix(self, lambdas=None):
        # TODO: make calling the memos more robust
        self._update_total_interpol_memos()
        if lambdas is None:
            lambdas = self.unique_lambdas
        M = self.calculateTotalStateEnergies(lambdas)
        return M

    def MBAR(self, recalculate=False, start=0., end=1., n_bootstraps=None, decorrelate=True, n_decorr=None, **kwargs):
        kwargs = {**self.pymbar_kwargs, **kwargs}
        key = (start, end)

        if recalculate or key not in self._fe_memo or self._fe_memo[key][0] >= self._last_total_interpol_memo_update:
            # initialise all data needed for MBAR
            lambdas = _np.asarray(self.unique_lambdas)
            u_kn = self.energyMatrix()
            N_k = self._total_interpol_memo["n"]

            # create decorrelated datasets if needed
            if decorrelate:
                tau = self.effectiveDecorrelationTime()
                max_distance = max(1, round(float(tau)))
                n_decorr = max_distance if n_decorr is None else max(1, min(n_decorr, max_distance))
                offsets = _np.random.choice(_np.arange(max_distance), size=n_decorr, replace=False)
                all_indices = [sorted(range(self._last_total_interpol_memo_update - 1 - x, -1, -max_distance)) for x in offsets]
                all_indices = self._time_to_mbar_indices(all_indices)
            else:
                all_indices = [_np.arange(u_kn.shape[1])]

            # get initial free energy estimates from previous calls to MBAR
            try:
                indices = self._time_to_mbar_indices([list(range(self._last_total_interpol_memo_update))])[0][:self._fe_memo[key][-1][0]._u_kn.shape[1]]
                u_kn_test = u_kn[:, _np.sort(indices)]
                old_fes = self._fe_memo[key][-1][0].computePerturbedFreeEnergies(u_kn_test, memo_key=tuple(lambdas))
                kwargs["initial_f_k"] = old_fes - old_fes[0]
            except (KeyError, AttributeError, TypeError):
                pass

            mbars = []
            for indices in all_indices:
                # constrain the indices to a particular lambda range
                min_lambda, max_lambda = _np.min(lambdas[lambdas >= start]), _np.max(lambdas[lambdas <= end])
                min_idx, max_idx = _np.where(lambdas == min_lambda)[0][0], _np.where(lambdas == max_lambda)[0][0]
                start_idx, end_idx = _np.sum(N_k[:min_idx]), _np.sum(N_k[:max_idx + 1])
                indices_constr = indices[(indices < end_idx) & (indices >= start_idx)]
                # bootstrap if needed
                if n_bootstraps is not None:
                    weights = self._total_interpol_memo["w"][indices]
                    N_resample = _np.round(_np.sum(weights)).astype(_np.int)
                    weights /= _np.sum(weights)
                    train_indices = _np.random.choice(indices, size=(n_bootstraps, N_resample), p=weights)
                else:
                    train_indices = [indices]

                # finally create MBAR models
                for x in train_indices:
                    x = x[_np.in1d(x, indices_constr)]
                    if x.size:
                        try:
                            mbars += [_MBARResult(u_kn, N_k, train_indices=x, **kwargs)]
                        except:
                            _logger.warning(f"Error while creating MBAR result: {_sys.exc_info()[0]}")
                            mbars += [None]
                    else:
                        mbars += [None]

            # cache the free energies
            u_kn_pert = self.energyMatrix(self.current_lambdas)
            for mbar in mbars:
                if mbar is not None:
                    mbar.computePerturbedFreeEnergies(u_kn_pert, memo_key=tuple(self.current_lambdas))

            # update cache
            effective_sample_size = self._last_total_interpol_memo_update
            if decorrelate:
                effective_sample_size //= max_distance
            increment = self.min_update_fe + round(effective_sample_size * (self.freq_update_fe))
            next_fe_calculation = self._last_total_interpol_memo_update + increment
            self._fe_memo[key] = next_fe_calculation, self.current_lambdas, mbars

        return self._fe_memo[key][1:]

    @property
    def fe_estimator(self):
        def func(lambda1, lambda0, **kwargs):
            kwargs_default = dict(decorrelate=self.decorrelate_fe, n_bootstraps=self.n_bootstraps_fe,
                                  n_decorr=self.n_decorr_fe)
            kwargs = {**kwargs_default, **kwargs}
            lambdas, mbars = self.MBAR(**kwargs)
            if not set(lambdas).issuperset({lambda0, lambda1}):
                self._fe_memo = {}
                kwargs["recalculate"] = True
                lambdas, mbars = self.MBAR(**kwargs)
            memo_key = tuple(lambdas)
            fes = []
            for mbar in mbars:
                if mbar is None:
                    fes += [_np.inf]
                    continue
                result = mbar.computePerturbedFreeEnergies(memo_key=memo_key)
                fes += [result[memo_key.index(lambda1)] - result[memo_key.index(lambda0)]]
            return _np.asarray(fes)

        return func

    @fe_estimator.setter
    def fe_estimator(self, val):
        pass

    def effectiveDecorrelationTime(self):
        lambdas = self.lambda_history[self.lambda_history.index(1):]
        n_lambdas = len(lambdas)
        min_lambda = [x for x in self.current_lambdas if x <= min(lambdas)][-1]
        if not len(lambdas[lambdas.index(min(lambdas)):]):
            return 0.
        max_lambda = [x for x in self.current_lambdas if x >= max(lambdas[lambdas.index(min(lambdas)):])][0]

        if self._decorr_memo is None or not set(self._decorr_memo[0]).issuperset({min_lambda, max_lambda}):
            mbar = self.MBAR(recalculate=True, decorrelate=False)[-1]
            transition_matrix = self.expectedTransitionMatrix(self.current_lambdas, mbar)
            self._decorr_memo = self.current_lambdas, transition_matrix

        costs = [1] * len(self._decorr_memo[0])
        tau_bwd = self.expectedTransitionTime(self._decorr_memo[0], lambda0=1, lambda1=min_lambda, costs=costs,
                                              target0=0, target1=1, transition_matrix=self._decorr_memo[1])
        tau_fwd = self.expectedTransitionTime(self._decorr_memo[0], lambda0=min_lambda, lambda1=max_lambda, costs=costs,
                                              target0=1, target1=0, transition_matrix=self._decorr_memo[1])
        tau_total = tau_bwd + tau_fwd

        if tau_total and not _np.isinf(tau_total):
            decorrelation_time = n_lambdas / (tau_total * max(self.cycles, 1))
            _logger.debug(f"Relative effective decorrelation time is {decorrelation_time}")
        else:
            decorrelation_time = 0.

        return decorrelation_time

    def expectedTransitionMatrix(self, lambdas, mbar):
        if len(lambdas) < 2:
            raise ValueError("Need a lambda array of at least two lambda values")
        lambdas = _np.asarray(lambdas)
        lambdas[lambdas < 0] = 0.
        lambdas[lambdas > 1] = 1.

        # calculate an averaged transition matrix if several MBAR models have been passed
        if isinstance(mbar, list):
            return _np.average([self.expectedTransitionMatrix(lambdas, x) for x in mbar], axis=0)

        # generate the energy distributions from the MBAR model
        u_kn = self.energyMatrix(lambdas)
        fes = mbar.computePerturbedFreeEnergies(u_kn)
        log_probabilities = -u_kn + fes.reshape(-1, 1)

        # generate the predicted transition matrix from the MBAR model
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
            s = _np.sum(T[i, :])
            if s > 1.:
                T[i, :] /= s
            else:
                T[i, i] = 1. - s

        return T

    def expectedTransitionTime(self, lambdas, *args, lambda0=0., lambda1=1., target0=1, target1=0, costs=None,
                               transition_matrix=None, **kwargs):
        lambdas = _np.asarray(lambdas)
        lambdas[lambdas < 0] = 0.
        lambdas[lambdas > 1] = 1.
        idx0, idx1 = _np.where(lambdas == lambda0)[0][0], _np.where(lambdas == lambda1)[0][0]
        if idx0 == idx1:
            return 0.
        i = max(0, 2 * idx0 - 1) if target0 == 0 else min(2 * idx0, 2 * lambdas.shape[0] - 3)
        j = max(0, 2 * idx1 - 1) if target1 == 0 else min(2 * idx1, 2 * lambdas.shape[0] - 3)

        if transition_matrix is None:
            transition_matrix = self.expectedTransitionMatrix(lambdas, *args, **kwargs)
        else:
            if not (transition_matrix.shape[0] == transition_matrix.shape[1] == 2 * (lambdas.size - 1)):
                raise ValueError("Invalid transition matrix shape supplied")
        transition_matrix_del = _np.delete(_np.delete(transition_matrix, i, axis=0), i, axis=1)
        if costs is None:
            costs = [self.decorrelationSteps(lambda_) for lambda_ in lambdas]
        costs = _np.asarray([costs[0]] + [x for x in costs[1:-1] for _ in range(2)] + [costs[-1]])
        identity = _np.identity(transition_matrix_del.shape[0])
        b = _np.delete(transition_matrix @ costs, i, axis=0)
        j = j if j < i else j - 1

        try:
            tau = _np.linalg.solve(identity - transition_matrix_del, b)[j]
            if tau < 0:
                tau = _np.inf
        except _np.linalg.LinAlgError:
            tau = _np.inf

        return tau

    def expectedRoundTripTime(self, *args, lambda0=0., lambda1=1., **kwargs):
        lambda0, lambda1 = min(lambda0, lambda1), max(lambda0, lambda1)
        tau_fwd = self.expectedTransitionTime(*args, lambda0=lambda0, lambda1=lambda1, target0=1, target1=0, **kwargs)
        tau_bwd = self.expectedTransitionTime(*args, lambda0=lambda1, lambda1=lambda0, target0=0, target1=1, **kwargs)
        tau_total = tau_bwd + tau_fwd
        return tau_total

    def _closest_protocol(self, n, fixed_lambdas=None):
        if fixed_lambdas is None:
            fixed_lambdas = []
        fixed_lambdas = sorted(set(fixed_lambdas) | {0., 1., self.lambda_})
        if n in self._protocol_memo.keys():
            protocol = self._protocol_memo[n]
        elif any(k for k in self._protocol_memo.keys() if k > n):
            protocol = self._protocol_memo[sorted(k for k in self._protocol_memo.keys() if k > n)[0]]
        elif any(k for k in self._protocol_memo.keys() if k < n):
            protocol = self._protocol_memo[sorted(k for k in self._protocol_memo.keys() if k < n)[-1]]
        else:
            protocol = self.current_lambdas
        protocol = _interp1d(_np.linspace(0., 1., num=len(protocol)), protocol)(_np.linspace(0., 1., num=n)).tolist()
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

        def convert_x(x):
            x = _np.asarray(x)
            try:
                x[x < x_start] = x_start
                x[x > x_end] = x_end
                x = interp(x)
                if self.significant_lambda_figures is not None:
                    x = _np.round(x, self.significant_lambda_figures)
                x = _np.sort(x.tolist() + internal_fixed_lambdas)
            except NameError:
                x = _np.asarray(internal_fixed_lambdas)
            return x

        def f(x):
            x = convert_x(x)
            y = self.expectedRoundTripTime(x, self.MBAR()[-1], lambda0=start, lambda1=end)
            return y

        N = max(n_opt, 0)
        N_total = N + len(fixed_lambdas)
        N_internal = N + len(internal_fixed_lambdas)

        if N == 0:
            protocol = fixed_lambdas
            fun = f([])
            success = True
        else:
            # interpolate from the closest protocol
            y_interp = _np.asarray(self._closest_protocol(N_total, fixed_lambdas=fixed_lambdas))
            x_interp = _np.linspace(0., 1., num=len(y_interp))
            interp = _interp1d(x_interp, y_interp)

            # define some initial parameters for the minimiser
            x_start, x_end = x_interp[y_interp == start][0], x_interp[y_interp == end][-1]
            x0 = x_interp[~_np.isin(y_interp, fixed_lambdas, assume_unique=True)]
            x0[x0 < x_start] = x_start + (x_end - x_start) / (N + 1)
            x0[x0 > x_end] = x_end - (x_end - x_start) / (N + 1)
            f0 = f(x0)
            kwargs = {**dict(verb_log=False, verb_log_expensive=False, bounds=([x_start] * N, [x_end] * N),
                             maxfevals=10 * N, tolfun=tol), **kwargs}
            # we bound in case of slight numerical overflows
            sigma0 = min((x_end - x_start) / (N_internal - 2), 0.99)

            stdout, stderr = _sys.stdout.write, _sys.stderr.write
            _sys.stdout.write, _sys.stderr.write = _logger.debug, _logger.error
            # minimise
            if x0.size == 1:
                val = _cma.fmin(lambda x: f(x[:1]), _np.asarray(list(x0) * 2), sigma0=sigma0, options=kwargs)
                protocol = [val[0][0]] if val is not None and val[1] < f0 else x0
            else:
                val = _cma.fmin(f, x0, sigma0=sigma0, options=kwargs)
                protocol = list(val[0]) if val is not None and val[1] < f0 else x0
            _sys.stdout.write, _sys.stderr.write = stdout, stderr

            fun = val[1] if val is not None and val[1] < f0 else f0
            success = (val is not None)
            protocol = convert_x(protocol)
        self._protocol_memo[N_total] = protocol
        return protocol, fun, success

    def _discrete_optimise_protocol(self, fixed_lambdas=None, start=0., end=1., **kwargs):
        fixed_lambdas = self._augment_fixed_lambdas(fixed_lambdas=fixed_lambdas, start=start, end=end)

        # minimise the lambda values given an integer number of them
        @_functools.lru_cache(maxsize=None)
        def optfunc(n):
            return self._continuous_optimise_protocol(n, fixed_lambdas=fixed_lambdas, start=start, end=end, **kwargs)

        # minimise the number of lambda windows
        N_adapt = len([x for x in self.current_lambdas if x not in fixed_lambdas])
        while True:
            protocol_list = [(N_adapt + i, *optfunc(N_adapt + i)) for i in [0, 1, -1]]
            min_protocol = min(protocol_list, key=lambda x: x[2])
            if min_protocol[0] == N_adapt or _np.isinf(min_protocol[2]) or _np.isnan(min_protocol[2]):
                protocol_curr, fun_curr, success = min_protocol[1:]
                break
            else:
                N_adapt = min_protocol[0]

        return protocol_curr, fun_curr, success

    def optimiseProtocol(self, start=None, end=None, **kwargs):
        effective_sample_size = len(self.lambda_history)
        if self._next_optimisation is None:
            self.current_lambdas = self.lambda_history[:self.lambda_history.index(1) + 1]
        elif self._next_optimisation <= len(self.lambda_history):
            start = 0 if start is None else start
            end = 1 if end is None else end

            self.MBAR(recalculate=True, n_bootstraps=self.n_bootstraps_opt, decorrelate=self.decorrelate_opt,
                      n_decorr=self.n_decorr_opt)
            if self.decorrelate_opt:
                tau = self.effectiveDecorrelationTime()
                max_distance = max(1, round(float(tau)))
                effective_sample_size = len(self.lambda_history) // max_distance

            protocol, fun, success = self._discrete_optimise_protocol(start=start, end=end, **kwargs)
            if _np.isinf(fun) or _np.isnan(fun) or not success:
                _logger.info("Optimisation failed, most likely due to undersampling. Retrying with reduced "
                             "optimisation range...")
                start, end = self.effective_lambda_bounds
                protocol, fun, success = self._discrete_optimise_protocol(start=start, end=end, **kwargs)

            if not _np.isinf(fun) and not _np.isnan(fun) and success:
                self.current_lambdas = protocol
                tau = fun * self.integrator.getStepSize().in_units_of(_unit.nanosecond)
                _logger.info(f"The optimal protocol after adaptation between lambda = {start} and lambda = {end} with "
                             f"an expected round trip time of {tau} is: {self.current_lambdas}")
                transition_matrix = self.expectedTransitionMatrix(self.current_lambdas, self.MBAR()[-1])
                self._decorr_memo = self.current_lambdas, transition_matrix
                _logger.debug(f"The corresponding transition matrix is: \n{_np.round(transition_matrix, 3)}")
                _logger.debug(f"Relative effective decorrelation time is {self.effectiveDecorrelationTime()}")
            else:
                _logger.info("Optimisation failed, most likely due to undersampling. Proceeding with current "
                             "protocol...")
            self._fe_memo = {}
        else:
            return

        increment = self.min_update_opt + round(effective_sample_size * (self.freq_opt))
        self._next_optimisation = len(self.lambda_history) + increment
        self._prev_optimisation_index = len(self.lambda_history)
        self._optimisation_memo += [self._prev_optimisation_index]

    def sample(self, *args, **kwargs):
        if self.parallel and 1 in self.lambda_history[:-1]:
            p1 = _threading.Thread(target=super().sample, args=args, kwargs=kwargs)
            p1.start()
            p2 = _threading.Thread(target=self.fe_estimator, args=(0, 1))
            p2.start()
            p1.join()
            p2.join()
        else:
            super().sample(*args, **kwargs)
        if 1 in self.lambda_history:
            self._update_interpol_memos()

    def runSingleIteration(self, *args, **kwargs):
        if self.adaptive_mode:
            minimum_value = 1. / self.significant_lambda_figures
            for key in ["default_dlambda", "minimum_dlambda", "maximum_dlambda", "target_lambda"]:
                if key in kwargs.keys():
                    kwargs[key] = _np.sign(kwargs[key]) * max(minimum_value, round(abs(kwargs[key]),
                                                                                   self.significant_lambda_figures))
            if "fixed_lambdas" not in kwargs.keys() or kwargs["fixed_lambdas"] is None:
                kwargs["fixed_lambdas"] = []
            else:
                kwargs["fixed_lambdas"] = sorted(set(kwargs["fixed_lambdas"]))
        super().runSingleIteration(*args, **kwargs)

    def _runPostAdaptiveIteration(self, optimise_kwargs=None, **kwargs):
        if optimise_kwargs is None:
            optimise_kwargs = {}
        self.optimiseProtocol(**optimise_kwargs)
        super()._runPostAdaptiveIteration(**kwargs)

    @staticmethod
    def generateAlchSystem(*args, **kwargs):
        if "alchemical_factory" in kwargs.keys():
            if kwargs["alchemical_factory"] is not _GaussianSCFactory:
                _warnings.warn("Alchemical factory not supported. Switching to "
                               "AbsoluteAlchemicalGaussianSoftcoreFactory...")
        kwargs["alchemical_factory"] = _GaussianSCFactory
        return super(GlobalAdaptiveCyclicSMCSampler, GlobalAdaptiveCyclicSMCSampler).generateAlchSystem(*args, **kwargs)
