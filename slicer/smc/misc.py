from cached_property import cached_property as _cached_property
import collections as _collections
import math as _math

import numpy as _np
import pandas as _pd

from slicer.interpolate import BatchLinearInterp as _BatchLinearInterp


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


class WalkerMemo:
    def __init__(self):
        self._walker_memo = []
        self._protocol_memo = []
        self._energy_memo = []

    @_cached_property
    def interpolator(self):
        return _BatchLinearInterp(self._protocol_memo, self._energy_memo)

    @_cached_property
    def iterations(self):
        return _np.asarray([w.iteration for w in self.relevant_walkers])

    @_cached_property
    def lambdas(self):
        return _np.asarray([w.lambda_ for w in self.relevant_walkers])

    @_cached_property
    def lambda_counts(self):
        return _np.unique(self.lambdas, return_counts=True)[-1]

    @_cached_property
    def mbar_indices(self):
        return _np.argsort(_np.concatenate([_np.where(self.lambdas == x)[0] for x in self.unique_lambdas]))

    @_cached_property
    def relevant_walkers(self):
        return [w for w in self._walker_memo if w.lambda_ is not None and w.transform is None]

    @_cached_property
    def timesteps(self):
        return self._sorted_unique_hashes.size

    @_cached_property
    def unique_lambdas(self):
        return _np.unique(self.lambdas)

    @property
    def walkers(self):
        return self._walker_memo

    @_cached_property
    def weights(self):
        counters = _collections.Counter(self._hashes)
        return _np.asarray([1. / counters[k] for k in self._hashes])

    @_cached_property
    def _hashes(self):
        return _np.asarray([hash((i, lambda_)) for i, lambda_ in zip(self.iterations, self.lambdas)])

    @_cached_property
    def _sorted_unique_hashes(self):
        # Returning sorted hashes with NumPy is more expensive
        return _pd.unique(self._hashes)

    def resetMemos(self):
        for key, value in self.__class__.__dict__.items():
            if isinstance(value, _cached_property):
                self.__dict__.pop(key, None)

    def removeWalkers(self, walkers):
        if walkers:
            indices = [i for i, w in enumerate(self._walker_memo) if w not in walkers]
            self._walker_memo = [self._walker_memo[i] for i in indices]
            self._protocol_memo = [self._protocol_memo[i] for i in indices if i < len(self._protocol_memo)]
            self._energy_memo = [self._energy_memo[i] for i in indices if i < len(self._energy_memo)]
            self.resetMemos()

    def updateWalkers(self, walkers):
        if walkers:
            self.removeWalkers(walkers)
            self._walker_memo += walkers
            for walker in self._walker_memo:
                if walker.state is not None and walker not in walkers:
                    walker.state = None
            self.resetMemos()

    def updateEnergies(self, ensemble, lambdas):
        walkers = self.relevant_walkers[len(self._protocol_memo):]
        self._protocol_memo += [lambdas] * len(walkers)
        self._energy_memo += ensemble.calculateStateEnergies(lambdas, walkers=walkers).T.tolist()

    def updateWalkersAndEnergies(self, walkers, ensemble, lambdas):
        self.updateWalkers(walkers)
        self.updateEnergies(ensemble, lambdas)

    def energyMatrix(self, lambdas=None):
        if lambdas is None:
            lambdas = self.unique_lambdas
        return self.interpolator(_np.asarray(lambdas))

    def walker_to_mbar_indices(self, indices):
        return indices[_np.argsort(self.mbar_indices[indices])]

    def time_to_walker_indices(self, indices):
        relevant_hashes = self._sorted_unique_hashes[indices]
        new_indices = _np.concatenate([_np.where(self._hashes == h)[0] for h in relevant_hashes])
        return new_indices

    def time_to_mbar_indices(self, indices):
        return self.walker_to_mbar_indices(self.time_to_walker_indices(indices))
