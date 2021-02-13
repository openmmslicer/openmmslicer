from mock import Mock
from threading import RLock

import numpy as np

from slicer.correlation_metrics import *
from slicer.fe_estimators import EnsembleMBAR
from slicer.samplers.misc import WalkerMemo


def test_effectiveDecorrelationTime():
    walker_memo = Mock()
    walker_memo.lock = RLock()
    walker_memo.lambdas = np.asarray([0., 1., 0.5, 0., 1.])
    walker_memo.unique_lambdas = np.asarray([0., 0.5, 1])
    walker_memo.mbar_indices = WalkerMemo.mbar_indices.__get__(walker_memo, WalkerMemo)
    walker_memo.energyMatrix.side_effect = lambda *args, **kwargs: np.asarray([
        [0.2, 0.4, 0.1, 0.8, 2],
        [0.3, 0.6, 0.5, 0.1, 3],
        [0.5, 0.8, 1, 0.2, 0.9],
    ])
    walker_memo.relevant_walkers = [None] * 5
    walker_memo.round_trips = 1
    walker_memo.timesteps = 5
    walker_memo.timestep_lambdas = walker_memo.lambdas
    walker_memo.time_to_walker_indices.side_effect = lambda x: x
    walker_memo.walker_to_mbar_indices.side_effect = lambda x: WalkerMemo.walker_to_mbar_indices(walker_memo, x)

    fe_estimator = EnsembleMBAR(walker_memo)
    fe_estimator.interval = EffectiveDecorrelationTime(fe_estimator, protocol=np.asarray([0, 0.5, 1]))
    assert np.isclose(fe_estimator._interval(), 4 / 6.220159220666085)

    walker_memo.timestep_lambdas = np.asarray([0., 0.5, 1., 0.8, 0.9, 0.8, 0.9])
    walker_memo.timesteps = len(walker_memo.timestep_lambdas)
    decorr = EffectiveDecorrelationTime(fe_estimator, protocol=np.asarray([0, 0.82, 0.88, 1]))
    assert decorr.min_lambda == 0.82
    assert decorr.max_lambda == 0.88

    walker_memo.timestep_lambdas = np.asarray([0., 0.8])
    walker_memo.timesteps = len(walker_memo.timestep_lambdas)
    assert decorr.min_lambda is None
    assert decorr.max_lambda is None

    walker_memo.timestep_lambdas = np.asarray([0., 0.8, 1.])
    walker_memo.timesteps = len(walker_memo.timestep_lambdas)
    assert decorr.min_lambda is None
    assert decorr.max_lambda is None

    walker_memo.timestep_lambdas = np.asarray([0., 0.8, 1., 0.9])
    walker_memo.timesteps = len(walker_memo.timestep_lambdas)
    assert decorr.min_lambda is None
    assert decorr.max_lambda is None

    walker_memo.timestep_lambdas = np.asarray([0., 0.8, 1., 0.8])
    walker_memo.timesteps = len(walker_memo.timestep_lambdas)
    assert decorr.min_lambda == 0.82
    assert decorr.max_lambda is None

    walker_memo.timestep_lambdas = np.asarray([0., 0.8, 1., 0.7, 0.8])
    walker_memo.timesteps = len(walker_memo.timestep_lambdas)
    assert decorr.min_lambda == 0.82
    assert decorr.max_lambda is None
