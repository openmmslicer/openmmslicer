import math as _math



class LinearAlchemicalFunction:
    def __init__(self, start, end, full_interpolation=True):
        self.boundaries = start, end
        self.full_interpolation = full_interpolation

    @property
    def boundaries(self):
        return self._start, self._end

    @property
    def start(self):
        return self._start

    @property
    def end(self):
        return self._end

    @boundaries.setter
    def boundaries(self, val):
        if not isinstance(val, tuple) or len(val) != 2 or val[0] >= val[1] or any(not 0 <= x <= 1 for x in val):
            raise ValueError("Need to supply a valid tuple with two ordered values between 0 and 1")
        self._start, self._end = val

    def __call__(self, x):
        if x <= self._start:
            return 0.
        elif x >= self._end:
            return 1.
        else:
            return (x - self._start) / (self._end - self._start)


class Walker:
    _reset_cache_triggers = ["state", "lambda_", "transform", "reporter_filename", "frame"]

    def __init__(self, name, state=None, lambda_=None, iteration=None, transform=None, reporter_filename=None,
                 frame=None, logW=None, **kwargs):
        self.name = name
        self.state = state
        self.lambda_ = lambda_
        self.iteration = iteration
        self.transform = transform
        self.reporter_filename = reporter_filename
        self.frame = frame
        self.logW = logW
        self._energy_cache = {}
        self.__dict__.update(kwargs)

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
