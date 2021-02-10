from mock import Mock, patch

import numpy as np

from slicer.protocol import *


@patch("slicer.transition_metrics.ExpectedRoundTripTime.expectedRoundTripTime")
def test_optimiseContinuous(mockExpectedRoundTripTime):
    mockExpectedRoundTripTime.side_effect = lambda x, *args, **kwargs: np.average((x - 0.44) ** 2)
    obj = Mock()
    obj._protocol_memo = {}
    obj._value = np.linspace(0., 0.3, num=3).tolist() + [1.]
    obj._initial_protocol_guess.side_effect = lambda *args, **kwargs: OptimisableProtocol._initial_protocol_guess(
        obj, *args, **kwargs)
    obj._augment_fixed_values.return_value = []
    obj.ensemble = Mock()
    obj.ensemble.lambda_= 0.
    obj.significant_figures = None
    x, y, success = OptimisableProtocol.optimiseContinuous(obj, 5, maxfevals=1000000, tol=1e-16)
    assert success
    assert np.allclose(x, 0.44)
    assert np.isclose(y, 0)

    obj._augment_fixed_values.return_value = [0., 0.33, 0.88, 1.0]
    x, y, success = OptimisableProtocol.optimiseContinuous(obj, 2, maxfevals=1000000, tol=1e-16)
    assert success
    assert len(x) == 6
    assert np.sum(np.isin(obj._augment_fixed_values.return_value, x))
    assert np.sum(np.isclose(x, 0.44)) == 2
    assert np.isclose(mockExpectedRoundTripTime(np.asarray(x)), y)

    x, y, success = OptimisableProtocol.optimiseContinuous(obj, 0, maxfevals=1000000, tol=1e-16)
    assert success
    assert len(x) == 4
    assert np.all(np.isin(obj._augment_fixed_values.return_value, x))
    assert np.isclose(mockExpectedRoundTripTime(np.asarray(x)), y)


def test_optimiseDiscrete():
    obj = Mock()
    obj._augment_fixed_values.return_value = []
    obj.optimiseContinuous.side_effect = lambda x, **kwargs: (int(x), int(x) ** 2 - 8 * int(x), True)

    obj._value = [0.]
    x, y, success = OptimisableProtocol.optimiseDiscrete(obj)
    assert success
    assert x == 4
    assert y == -16

    obj._value = [0.] * 10
    x, y, success = OptimisableProtocol.optimiseDiscrete(obj)
    assert success
    assert x == 4
    assert y == -16
