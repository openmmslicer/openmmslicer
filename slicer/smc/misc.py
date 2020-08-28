import math as _math

import anytree as _anytree


class Walker(_anytree.Node):
    _reset_cache_triggers = ["state", "lambda_", "transform", "reporter_filename", "frame"]

    def __init__(self, *args, state=None, lambda_=None, iteration=None, transform=None, reporter_filename=None,
                 frame=None, logW=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.state = state
        self.lambda_ = lambda_
        self.iteration = iteration
        self.transform = transform
        self.reporter_filename = reporter_filename
        self.frame = frame
        self.logW = logW
        self._energy_cache = {}

    def __setattr__(self, key, value):
        if key in self._reset_cache_triggers:
            self.resetCache()
        super().__setattr__(key, value)

    def getCachedEnergy(self, lambda_):
        energies = [x for x in self._energy_cache.keys() if _math.isclose(x, lambda_)]
        if not len(energies):
            return None
        else:
            return self._energy_cache[energies[0]]

    def setCachedEnergy(self, lambda_, energy):
        keys = [x for x in self._energy_cache.keys() if _math.isclose(x, lambda_)]
        if not len(keys):
            key = lambda_
        else:
            key = keys[0]
        self._energy_cache[key] = energy

    def setStateKeepCache(self, state):
        """This sets a state whilst keeping the energy cache. This will result in wrong energy values once the state
        changes, so use with caution."""
        super().__setattr__("state", state)

    def resetCache(self):
        self._energy_cache = {}
