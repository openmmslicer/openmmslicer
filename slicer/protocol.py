import functools as _functools
import logging as _logging

import cma as _cma
import numpy as _np
from scipy.interpolate import interp1d as _interp1d
from simtk import unit as _unit
import stdio_proxy as _stdio_proxy

import slicer.misc as _misc
from slicer.transition_metrics import ExpectedRoundTripTime as _ExpectedRoundTripTime

_logger = _logging.getLogger(__name__)


class Protocol:
    def __init__(self, protocol, fixed_values=None, significant_figures=None):
        self.significant_figures = significant_figures
        self.fixed_values = fixed_values
        self.value = protocol

    @property
    def fixed_values(self):
        return self._fixed_values

    @fixed_values.setter
    def fixed_values(self, val):
        if val is None or not len(val):
            val = []
        self._fixed_values = self._round(val)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        if val is None or not len(val):
            val = []
        self._value = self._round(sorted(set(val) | set(self.fixed_values)))

    def _round(self, val):
        if self.significant_figures is not None:
            return _np.round(val, self.significant_figures)
        else:
            return _np.asarray(val)


class OptimisableProtocol(Protocol):
    def __init__(self, ensemble, *args,
                 update_func=lambda self: 100 + 0.1 * self.ensemble.fe_estimator.effective_sample_size, **kwargs):
        self.update_func = update_func
        self.ensemble = ensemble
        self._protocol_memo = {}
        self._prev_optimisation = self.ensemble.fe_estimator.walker_memo.timesteps
        self._next_optimisation = self._prev_optimisation + self.update_func(self)
        super().__init__(*args, **kwargs)

    @Protocol.fixed_values.getter
    def fixed_values(self):
        return _np.asarray(sorted(set(self._fixed_values) | {self._round(self.ensemble.lambda_)}))

    @Protocol.value.getter
    def value(self):
        if self._next_optimisation <= self.ensemble.fe_estimator.walker_memo.timesteps:
            self.optimise()
        return super().value

    def _augment_fixed_values(self, fixed_values=None, start=0., end=1.):
        if fixed_values is None:
            fixed_values = self.fixed_values
        fixed_values = set(fixed_values) | {x for x in self._value if not start < x < end} | {self.ensemble.lambda_}
        return _np.asarray(sorted(fixed_values))

    def _initial_protocol_guess(self, n, fixed_values=None):
        if fixed_values is None:
            fixed_values = self.fixed_values
        if n in self._protocol_memo.keys():
            protocol = self._protocol_memo[n]
        elif any(k for k in self._protocol_memo.keys() if k > n):
            protocol = self._protocol_memo[sorted(k for k in self._protocol_memo.keys() if k > n)[0]]
        elif any(k for k in self._protocol_memo.keys() if k < n):
            protocol = self._protocol_memo[sorted(k for k in self._protocol_memo.keys() if k < n)[-1]]
        else:
            protocol = self._value
        protocol = list(_interp1d(_np.linspace(0., 1., num=len(protocol)), protocol)(_np.linspace(0., 1., num=n)))
        for lambda_ in fixed_values:
            protocol.remove(min(protocol, key=lambda x: abs(x - lambda_)))
        return _np.sort(protocol + fixed_values)

    @property
    def effective_lambda_bounds(self):
        lambdas_recent = self.ensemble.fe_estimator.walker_memo[self._prev_optimisation:]
        if not lambdas_recent.size:
            min_lambda = max_lambda = self.ensemble.lambda_
        else:
            min_lambda, max_lambda = min(lambdas_recent), max(lambdas_recent)
        min_lambda_idx = _np.where(self._value == min_lambda)[0][0]
        max_lambda_idx = _np.where(self._value == max_lambda)[0][0]
        left = self._value[max(0, min_lambda_idx - 1)]
        right = self._value[min(max_lambda_idx + 1, len(self._value) - 1)]
        return left, right

    def optimiseContinuous(self, n_opt, fixed_values=None, start=0., end=1., tol=1e-3, **kwargs):
        if fixed_values is None:
            fixed_values = self.fixed_values
        fixed_values = list(self._augment_fixed_values(fixed_values=fixed_values, start=start, end=end))
        internal_fixed_values = [x for x in fixed_values if start <= x <= end]

        def convert_x(x):
            x = _np.asarray(x)
            try:
                x[x < x_start] = x_start
                x[x > x_end] = x_end
                x = interp(x)
                if self.significant_figures is not None:
                    x = _np.round(x, self.significant_figures)
                x = _np.sort(list(x) + internal_fixed_values)
            except NameError:
                x = _np.asarray(internal_fixed_values)
            return x

        def f(x):
            x = convert_x(x)
            costs = self.ensemble.decorrelationSteps(x)
            y = _ExpectedRoundTripTime(self.ensemble.fe_estimator).expectedRoundTripTime(x, lambda0=start, lambda1=end,
                                                                                         costs=costs)
            return y

        N = max(n_opt, 0)
        N_total = N + len(fixed_values)
        N_internal = N + len(internal_fixed_values)

        if N == 0:
            protocol = fixed_values
            fun = f([])
            success = True
        else:
            # interpolate from the closest protocol
            y_interp = _np.asarray(self._initial_protocol_guess(N_total, fixed_values=fixed_values))
            x_interp = _np.linspace(0., 1., num=len(y_interp))
            interp = _interp1d(x_interp, y_interp)

            # define some initial parameters for the minimiser
            x_start, x_end = x_interp[y_interp == start][0], x_interp[y_interp == end][-1]
            x0 = x_interp[~_np.isin(y_interp, fixed_values, assume_unique=True)]
            x0[x0 < x_start] = x_start + (x_end - x_start) / (N + 1)
            x0[x0 > x_end] = x_end - (x_end - x_start) / (N + 1)
            f0 = f(x0)
            kwargs = {**dict(verb_log=False, verb_log_expensive=False, bounds=([x_start] * N, [x_end] * N),
                             maxfevals=10 * N, tolfun=tol), **kwargs}

            # we bound in case of slight numerical overflows
            sigma0 = min((x_end - x_start) / (N_internal - 2), 0.99)

            # minimise
            with _stdio_proxy.redirect_stdout(_misc.LoggerWriter(_logger.debug)):
                if x0.size == 1:
                    val = _cma.fmin(lambda x: f(x[:1]), _np.asarray(list(x0) * 2), sigma0=sigma0, options=kwargs)
                    protocol = [val[0][0]] if val is not None and val[1] < f0 else x0
                else:
                    val = _cma.fmin(f, x0, sigma0=sigma0, options=kwargs)
                    protocol = list(val[0]) if val is not None and val[1] < f0 else x0

            fun = val[1] if val is not None and val[1] < f0 else f0
            success = (val is not None)
            protocol = convert_x(protocol)
        self._protocol_memo[N_total] = protocol
        return protocol, fun, success

    def optimiseDiscrete(self, fixed_values=None, start=0., end=1., **kwargs):
        if fixed_values is None:
            fixed_values = self.fixed_values
        fixed_values = self._augment_fixed_values(fixed_values=fixed_values, start=start, end=end)

        # minimise the lambda values given an integer number of them
        @_functools.lru_cache(maxsize=None)
        def optfunc(n):
            return self.optimiseContinuous(n, fixed_values=fixed_values, start=start, end=end, **kwargs)

        # minimise the number of lambda windows
        N_adapt = len([x for x in self._value if x not in fixed_values])
        while True:
            protocol_list = [(N_adapt + i, *optfunc(N_adapt + i)) for i in [0, 1, -1]]
            min_protocol = min(protocol_list, key=lambda x: x[2])
            if min_protocol[0] == N_adapt or _np.isinf(min_protocol[2]) or _np.isnan(min_protocol[2]):
                protocol_curr, fun_curr, success = min_protocol[1:]
                break
            else:
                N_adapt = min_protocol[0]

        return protocol_curr, fun_curr, success

    @_misc.norecurse()
    def optimise(self, start=0., end=1., **kwargs):
        protocol, fun, success = self.optimiseDiscrete(start=start, end=end, **kwargs)
        if _np.isinf(fun) or _np.isnan(fun) or not success:
            start_eff, end_eff = self.effective_lambda_bounds
            if start_eff < start or end_eff > end:
                _logger.info("Optimisation failed, most likely due to undersampling. Retrying with reduced "
                             "optimisation range...")
                start = min(start, start_eff)
                end = max(end, end_eff)
                protocol, fun, success = self.optimiseDiscrete(start=start, end=end, **kwargs)

        if not _np.isinf(fun) and not _np.isnan(fun) and success:
            self.value = protocol
            tau = fun * self.ensemble.integrator.getStepSize().in_units_of(_unit.nanosecond)
            _logger.info(f"The optimal protocol after adaptation between lambda = {start} and lambda = {end} with "
                         f"an expected round trip time of {tau} is: {self._value}")
        else:
            _logger.info("Optimisation failed, most likely due to undersampling. Proceeding with current "
                         "protocol...")
        self._prev_optimisation = self.ensemble.fe_estimator.walker_memo.timesteps
        self._next_optimisation = self._prev_optimisation + self.update_func(self)
