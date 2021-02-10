import numpy as np

from slicer.samplers.misc import *


def test_WalkerMemo():
    walkers = [Walker(i, lambda_=lambda_, iteration=iteration) for i, (lambda_, iteration) in
               enumerate(zip([0, 1, 0.5, 0.5, 0.5, 1, 0, 1], [0, 0, 0, 0, 0, 1, 1, 2]))]
    walker_memo = WalkerMemo()
    walker_memo.updateWalkers(walkers)

    expected_unique_lambdas = np.asarray([0, 0.5, 1])
    assert np.array_equal(walker_memo.unique_lambdas, expected_unique_lambdas)
    expected_timestep_lambdas = np.asarray([0, 1, 0.5, 1, 0, 1])
    assert np.array_equal(walker_memo.timestep_lambdas, expected_timestep_lambdas)
    expected_mbar_indices = np.asarray([0, 5, 2, 3, 4, 6, 1, 7])
    assert np.array_equal(walker_memo.mbar_indices, expected_mbar_indices)

    time_indices = np.asarray([0, 2, 4, 5])
    expected_walker_indices = np.asarray([0, 2, 3, 4, 6, 7])
    assert np.array_equal(walker_memo.time_to_walker_indices(time_indices), expected_walker_indices)
    expected_mbar_indices = np.asarray([0, 6, 2, 3, 4, 7])
    assert np.array_equal(walker_memo.walker_to_mbar_indices(expected_walker_indices), expected_mbar_indices)
    assert np.array_equal(walker_memo.time_to_mbar_indices(time_indices), expected_mbar_indices)

    walker_indices = np.asarray([3, 3, 6, 5, 0, 2])
    expected_mbar_indices = np.asarray([0, 6, 2, 3, 3, 5])
    assert np.array_equal(walker_memo.walker_to_mbar_indices(walker_indices), expected_mbar_indices)
