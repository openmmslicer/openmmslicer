from math import isclose
from mock import Mock

import numpy as np

from slicer.smc.global_adaptive import GlobalAdaptiveCyclicSMCSampler


def test_expectedRoundTripTime():
    # test described here: http://dx.doi.org/10.1080/0020739950260510
    obj = Mock()
    obj.decorrelationSteps.return_value = 1.
    obj.expectedTransitionMatrix.return_value = np.asarray([
        [.3, .1, .4, .2],
        [.2, .5, .2, .1],
        [.3, .2, .1, .4],
        [0., .6, .3, .1],
    ])
    assert isclose(GlobalAdaptiveCyclicSMCSampler.expectedRoundTripTime(obj, [0.] * 3), 10.646234070290262)
    obj.expectedTransitionMatrix.return_value = obj.expectedTransitionMatrix()[::-1, ::-1]
    assert isclose(GlobalAdaptiveCyclicSMCSampler.expectedRoundTripTime(obj, [0.] * 3), 10.646234070290262)


def test_continuous_optimise_protocol():
    obj = Mock()
    obj._protocol_memo = {}
    obj.current_lambdas = np.linspace(0., 0.3, num=3).tolist() + [1.]
    obj.expectedRoundTripTime.side_effect = lambda x, *args, **kwargs: np.average((x - 0.44) ** 2)
    obj.MBAR.return_value = [(None, None)]
    obj._closest_protocol.side_effect = lambda *args, **kwargs: GlobalAdaptiveCyclicSMCSampler._closest_protocol(
        obj, *args, **kwargs)
    obj._augment_fixed_lambdas.return_value = []
    obj.lambda_ = 0.
    x, y = GlobalAdaptiveCyclicSMCSampler._continuous_optimise_protocol(obj, 5, maxfevals=1000000, tol=1e-16)
    assert np.sum(np.isclose(x, 0.44)) == 5
    assert np.isclose(y, 0)

    obj._augment_fixed_lambdas.return_value = [0., 0.33, 0.88, 1.0]
    x, y = GlobalAdaptiveCyclicSMCSampler._continuous_optimise_protocol(obj, 2, maxfevals=1000000, tol=1e-16)
    assert len(x) == 6
    assert np.sum(np.isin(x, obj._augment_fixed_lambdas.return_value)) == 4
    assert np.sum(np.isclose(x, 0.44)) == 2
    assert np.isclose(obj.expectedRoundTripTime.side_effect(np.asarray(x)), y)

    x, y = GlobalAdaptiveCyclicSMCSampler._continuous_optimise_protocol(obj, 0, maxfevals=1000000, tol=1e-16)
    assert len(x) == 4
    assert np.sum(np.isin(x, obj._augment_fixed_lambdas.return_value)) == 4
    assert np.isclose(obj.expectedRoundTripTime.side_effect(np.asarray(x)), y)


def test_discrete_optimise_protocol():
    obj = Mock()
    obj._augment_fixed_lambdas.return_value = []
    obj._continuous_optimise_protocol.side_effect = lambda x, **kwargs: (int(x), int(x) ** 2 - 8 * int(x))

    obj.current_lambdas = [0.]
    x, y = GlobalAdaptiveCyclicSMCSampler._discrete_optimise_protocol(obj)
    assert x == 4
    assert y == -16

    obj.current_lambdas = [0.] * 10
    x, y = GlobalAdaptiveCyclicSMCSampler._discrete_optimise_protocol(obj)
    assert x == 4
    assert y == -16
