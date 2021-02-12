import logging as _logging

import numpy as _np

from .st import STSampler as _STSampler
from .misc import LinearAlchemicalFunction as _LinearAlchemicalFunction
from slicer.protocol import OptimisableProtocol as _OptimisableProtocol

_logger = _logging.getLogger(__name__)


class FASTSampler(_STSampler):
    # TODO: implement threading
    default_alchemical_functions = {
        'lambda_sterics': _LinearAlchemicalFunction(0, 0.5),
        'lambda_electrostatics': _LinearAlchemicalFunction(0.5, 1),
        'lambda_torsions': _LinearAlchemicalFunction(0, 0.5),
    }

    def __init__(self, *args, significant_figures=2,
                 prot_update_func=lambda self: 100 + 0.1 * self.ensemble.fe_estimator.effective_sample_size, **kwargs):
        super().__init__(*args, **kwargs)
        self._post_adaptive_kwargs.update(dict(significant_figures=significant_figures,
                                               prot_update_func=prot_update_func))

    @_STSampler.alchemical_functions.setter
    def alchemical_functions(self, val):
        if val is None:
            val = {}
        for key, value in val.items():
            if not isinstance(value, _LinearAlchemicalFunction):
                raise TypeError(f"Value {value} must be an instance of LinearAlchemicalFunction")

        self._alchemical_functions = {**self.default_alchemical_functions, **val}
        # this shouldn't be needed but better safe than sorry
        for func in self.alchemical_functions.values():
            assert func(0) == 0 and func(1) == 1, "All alchemical functions must go from 0 to 1"

    @property
    def interpolatable_protocol(self):
        lambdas = set()
        for key, value in self.alchemical_functions.items():
            if value.full_interpolation:
                lambdas |= {x for x in self.protocol.value if value.start <= x <= value.end}
            lambdas |= set(value.boundaries)
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
                                             significant_figures=self._post_adaptive_kwargs["significant_figures"],
                                             update_func=self._post_adaptive_kwargs["prot_update_func"])
        self.walker_memo.updateEnergies(self, self.protocol.value)
