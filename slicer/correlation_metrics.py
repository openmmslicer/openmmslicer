import inspect as _inspect

import numpy as _np

from slicer.transition_metrics import ExpectedRoundTripTime as _ExpectedRoundTripTime


class EffectiveDecorrelationTime:
    def __init__(self, fe_estimator=None, protocol=None):
        self.fe_estimator = fe_estimator
        self.protocol = protocol
        self._last_value = None

    @property
    def max_lambda(self):
        if self.min_lambda is not None:
            idx = _np.where(self.fe_estimator.walker_memo.timestep_lambdas == self.min_lambda)[0]
            if idx[0] < self.fe_estimator.walker_memo.timestep_lambdas.size:
                return _np.max(self.fe_estimator.walker_memo.timestep_lambdas[idx[0]:])
        return None

    @property
    def min_lambda(self):
        if self.fe_estimator.walker_memo.timesteps:
            idx = _np.where(self.fe_estimator.walker_memo.timestep_lambdas == 1.)[0]
            if idx.size and idx[0] < self.fe_estimator.walker_memo.timesteps:
                return _np.min(self.fe_estimator.walker_memo.timestep_lambdas[idx[0]:])
        return None

    def __call__(self):
        frames = _inspect.getouterframes(_inspect.currentframe())[1:]
        recurse = any(x for x in frames if x.function == '__call__' and "self" in x.frame.f_locals.keys()
                      and x.frame.f_locals["self"] is self)
        if not recurse:
            if self.min_lambda is None or self.max_lambda is None:
                self._last_value = None
            else:
                model = _ExpectedRoundTripTime(self.fe_estimator)
                tau_bwd = model.expectedTransitionTime(self.protocol, lambda0=1, lambda1=self.min_lambda, target0=0,
                                                       target1=1)
                tau_fwd = model.expectedTransitionTime(self.protocol, lambda0=self.min_lambda, lambda1=self.max_lambda,
                                                       target0=1, target1=0)
                tau_total = tau_bwd + tau_fwd

                if tau_total and not _np.isinf(tau_total) and not _np.isnan(tau_total):
                    n_lambdas = self.fe_estimator.walker_memo.timesteps - \
                                _np.where(self.fe_estimator.walker_memo.timestep_lambdas == 1.)[0][0]
                    self._last_value = n_lambdas / (tau_total * max(self.fe_estimator.walker_memo.round_trips, 1))
                else:
                    self._last_value = None
        return self._last_value
