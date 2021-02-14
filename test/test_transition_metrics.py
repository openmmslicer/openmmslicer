from mock import Mock
from threading import RLock

import numpy as np

from slicer.fe_estimators import MBAR
from slicer.samplers.misc import WalkerMemo
from slicer.transition_metrics import *


def test_expectedTransitionMatrix():
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
    walker_memo.timesteps = 5
    walker_memo.time_to_walker_indices.side_effect = lambda x: x
    walker_memo.walker_to_mbar_indices.side_effect = lambda x: WalkerMemo.walker_to_mbar_indices(walker_memo, x)

    for parallel in [False, True]:
        fe_estimator = MBAR(walker_memo, parallel=parallel)

        expected_transition_matrix = np.asarray([
            [0.16397161, 0.00000000, 0.83602839, 0.00000000],
            [0.83602839, 0.06815437, 0.09581724, 0.00000000],
            [0.00000000, 0.23354467, 0.06815437, 0.69830096],
            [0.00000000, 0.69830096, 0.00000000, 0.30169904],
        ])
        transition_matrix = ExpectedRoundTripTime(fe_estimator).expectedTransitionMatrix([0, 0.5, 1])
        assert np.allclose(transition_matrix, expected_transition_matrix, rtol=1e-8)


def test_expectedRoundTripTime():
    # test described here: http://dx.doi.org/10.1080/0020739950260510
    obj = Mock()
    obj.expectedTransitionMatrix.return_value = np.asarray([
        [.3, .1, .4, .2],
        [.2, .5, .2, .1],
        [.3, .2, .1, .4],
        [0., .6, .3, .1],
    ])
    obj._cast.side_effect = lambda *args, **kwargs: ExpectedRoundTripTime._cast(*args, **kwargs)
    obj.expectedTransitionTime.side_effect = lambda *args, **kwargs: \
        ExpectedRoundTripTime.expectedTransitionTime(obj, *args, **kwargs)
    assert np.isclose(ExpectedRoundTripTime.expectedRoundTripTime(obj, [0, 0.5, 1]), 10.646234070290262)
    obj.expectedTransitionMatrix.return_value = obj.expectedTransitionMatrix()[::-1, ::-1]
    assert np.isclose(ExpectedRoundTripTime.expectedRoundTripTime(obj, [0, 0.5, 1]), 10.646234070290262)
