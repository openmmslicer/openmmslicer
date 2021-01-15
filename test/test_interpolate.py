import numpy as np

from slicer.interpolate import BatchLinearInterp


def test_LinearInterp():
    xs = [[0, 1, 0.5], [0.2, 1, 0]]
    ys = [[10, 20, 11], [300, 1000, 200]]
    x_interp = [0.25, 0.5]
    result_correct = np.asarray([[10.5, 343.75], [11., 562.5]])
    result_obs = BatchLinearInterp(xs, ys)(x_interp)
    assert np.sum(np.abs(result_correct - result_obs) < 1e-8) == 4

    xs = [[0, 1, 0.5], [0.4, 1, 0, 0.6]]
    ys = [[10, 20, 11], [300, 1000, 200, 500]]
    x_interp = [0.25, 0.5]
    result_correct = np.asarray([[10.5, 262.5], [11., 400]])
    result_obs = BatchLinearInterp(xs, ys)(x_interp)
    assert np.sum(np.abs(result_correct - result_obs) < 1e-8) == 4