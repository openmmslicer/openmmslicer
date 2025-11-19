import logging as _logging

import numpy as _np

from .st import STSampler as _STSampler
from .misc import LinearAlchemicalFunction as _LinearAlchemicalFunction
import openmmslicer.fe_estimators as _fe_estimators
from openmmslicer.protocol import OptimisableProtocol as _OptimisableProtocol

_logger = _logging.getLogger(__name__)


class FASTSampler(_STSampler):
    """
    An irreversible simulated tempering sampler.

    Parameters
    ----------
    prot_update_func : callable, optional
        A callable which when called as prot_update_func(protocol) returns the number of times sample() will be called
        before the next protocol update is performed. Default is
        100 + 0.1 * self.ensemble.fe_estimator.effective_sample_size.
    args
        Positional arguments to be passed to super().__init__().
    kwargs
        Keyword arguments to be passed to super().__init__().

    Attributes
    ----------
    interpolatable_protocol : numpy.ndarray
        The sequence of lambda values used for energy interpolation.
    """
    picklable_attrs = _STSampler._picklable_attrs + [
        "_post_adaptive_kwargs"
    ]

    default_alchemical_functions = {
        'lambda_bonds': _LinearAlchemicalFunction(0, 0.5),
        'lambda_angles': _LinearAlchemicalFunction(0, 0.5),
        'lambda_electrostatics': _LinearAlchemicalFunction(0.5, 1),
        'lambda_sterics': _LinearAlchemicalFunction(0, 0.5),
        'lambda_torsions': _LinearAlchemicalFunction(0, 0.5),
    }

    def __init__(self, *args, alchemical_functions=None, fe_estimator=_fe_estimators.MBAR, significant_figures=2,
                 prot_update_func=lambda self: 100 + 0.1 * self.ensemble.fe_estimator.effective_sample_size,
                 checkpoint=None, **kwargs):
        if alchemical_functions is None:
            alchemical_functions = {}
        for key, value in alchemical_functions.items():
            if not isinstance(value, _LinearAlchemicalFunction):
                raise TypeError(f"Value {value} must be an instance of LinearAlchemicalFunction")
        alchemical_functions = {**self.default_alchemical_functions, **alchemical_functions}
        super().__init__(*args, alchemical_functions=alchemical_functions, fe_estimator=fe_estimator, **kwargs)
        self._post_adaptive_kwargs.update(dict(significant_figures=significant_figures,
                                               prot_update_func=prot_update_func))
        if checkpoint is not None:
            self.loadCheckpoint(checkpoint)

    @property
    def interpolatable_protocol(self):
        lambdas = set()
        n_states = len(self.alchemical_chain.states) - 1
        for name, func in self.simulation.alchemical_functions.items():
            for i in range(n_states):
                if func.full_interpolation:
                    lambdas |= {x for x in self.protocol.value
                                if (i + func.start) / n_states <= x <= (i + func.end) / n_states}
                lambdas |= {(i + x) / n_states for x in func.boundaries}
        return _np.asarray(sorted(lambdas))

    @_STSampler.walkers.setter
    def walkers(self, val):
        if self.protocol is None:
            self.walker_memo.updateWalkers(val)
        else:
            self.walker_memo.updateWalkersAndEnergies(val, self, self.interpolatable_protocol)
        self._walkers = val

    def _initialise_protocol(self):
        idx = _np.where(self.walker_memo.timestep_lambdas == 1)[0][0]
        self.protocol = _OptimisableProtocol(self, self.walker_memo.timestep_lambdas[:idx + 1],
                                             fixed_values=self.alchemical_chain.fixed_lambdas,
                                             significant_figures=self._post_adaptive_kwargs["significant_figures"],
                                             update_func=self._post_adaptive_kwargs["prot_update_func"])
        self.walker_memo.updateEnergies(self, self.protocol.value)

    def loadCheckpoint(self, *args, **kwargs):
        super().loadCheckpoint(*args, **kwargs)
        self._protocol.ensemble = self

    def serialise(self):
        pickle_dict = super().serialise()
        pickle_dict["_protocol"].ensemble = None
        return pickle_dict