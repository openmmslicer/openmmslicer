import logging as _logging

import numpy as _np

from openmmslicer.misc import norecurse as _norecurse
from openmmslicer.protocol import Protocol as _Protocol
from openmmslicer.transition_metrics import ExpectedRoundTripTime as _ExpectedRoundTripTime

_logger = _logging.getLogger(__name__)


class EffectiveDecorrelationTime:
    """
    Calculates an effective decorrelation time of a sequence of walkers using the ratio between the observed round-trip
    time and the expected round-trip time calculated using openmmslicer.transition_metrics.ExpectedRoundTripTime.

    Parameters
    ----------
    fe_estimator : openmmslicer.fe_estimators.AbstractFEEstimator, optional
        The free energy estimator used to estimate the free energies between the lambda windows.
    protocol : openmmslicer.protocols.Protocol, optional
        The protocol used to estimate the round-trip time.

    Attributes
    ----------
    min_lambda
    max_lambda
    protocol
    fe_estimator : openmmslicer.fe_estimators.AbstractFEEstimator
        The free energy estimator.
    """
    def __init__(self, fe_estimator=None, protocol=None):
        self.fe_estimator = fe_estimator
        self.protocol = protocol
        self._last_value = None
        self._last_update = None

    @property
    def max_lambda(self):
        """float or None : The largest lambda value reached after self.min_lambda."""
        if self.min_lambda is not None:
            idx_1 = _np.where(self.fe_estimator.walker_memo.timestep_lambdas == 1.)[0]
            if idx_1.size:
                min_lambda = _np.min(self.fe_estimator.walker_memo.timestep_lambdas[idx_1[0]:])
                idx_min = _np.where(self.fe_estimator.walker_memo.timestep_lambdas == min_lambda)[0]
                idx_min = _np.min(idx_min[idx_min > idx_1[0]])
                if idx_min < self.fe_estimator.walker_memo.timesteps - 1:
                    val = _np.max(self.fe_estimator.walker_memo.timestep_lambdas[idx_min:])
                    if self.protocol is not None:
                        val = _np.max(self.protocol[self.protocol <= val])
                        if val <= self.min_lambda:
                            return None
                    return float(val)
        return None

    @property
    def min_lambda(self):
        """float or None : The smallest lambda value reached after the first occurrence of lambda = 1."""
        if self.fe_estimator.walker_memo.timesteps:
            idx = _np.where(self.fe_estimator.walker_memo.timestep_lambdas == 1.)[0]
            if idx.size:
                val = _np.min(self.fe_estimator.walker_memo.timestep_lambdas[idx[0]:])
                if self.protocol is not None:
                    val = _np.min(self.protocol[self.protocol >= val])
                    if val == 1:
                        return None
                return float(val)
        return None

    @property
    def protocol(self):
        """openmm.protocols.Protocol : The protocol."""
        if isinstance(self._protocol, _Protocol):
            return self._protocol.value
        else:
            return self._protocol

    @protocol.setter
    def protocol(self, val):
        self._protocol = val

    @_norecurse(default_return_value=lambda self: self._last_value)
    def __call__(self):
        """float : The relative effective decorrelation time."""
        if self._last_update is not None and self._last_update == self.fe_estimator.walker_memo.timesteps:
            return self._last_value
        self._last_update = self.fe_estimator.walker_memo.timesteps
        if self.protocol is not None:
            if self.min_lambda is None:
                self._last_value = None
            else:
                model = _ExpectedRoundTripTime(self.fe_estimator)
                tau = model.expectedTransitionTime(self.protocol, lambda0=1, lambda1=self.min_lambda, target0=0,
                                                   target1=1)
                if self.max_lambda is not None:
                    tau += model.expectedTransitionTime(self.protocol, lambda0=self.min_lambda, lambda1=self.max_lambda,
                                                        target0=1, target1=0)

                # TODO: use actual sampling time instead
                if tau and not _np.isinf(tau) and not _np.isnan(tau):
                    n_lambdas = self.fe_estimator.walker_memo.timesteps - \
                                _np.where(self.fe_estimator.walker_memo.timestep_lambdas == 1.)[0][0]
                    self._last_value = n_lambdas / (tau * max(self.fe_estimator.walker_memo.round_trips, 1))
                    _logger.debug(f"Relative effective decorrelation time is {self._last_value}")
                else:
                    self._last_value = None
        return self._last_value
