from mock import Mock
import numpy as np

from slicer.samplers.misc import WalkerMemo
from slicer.fe_estimators import MBARResult, MBAR, BAR, EnsembleMBAR, EnsembleBAR

walker_memo = Mock()
walker_memo.lambdas = np.asarray([0., 1., 0., 0., 1.])
walker_memo.unique_lambdas = np.asarray([0., 1])
walker_memo.mbar_indices = WalkerMemo.mbar_indices.__get__(walker_memo, WalkerMemo)
walker_memo.energyMatrix.side_effect = lambda *args, **kwargs: np.asarray([
    [0.2, 1.0, 0.5, 0.7, -2.3],
    [30.5, 30.1, 42.5, 880.2, 33.6]
])
walker_memo.relevant_walkers = [None] * 5
walker_memo.timesteps = 5
walker_memo.time_to_walker_indices.side_effect = lambda x: x
walker_memo.walker_to_mbar_indices.side_effect = lambda x: WalkerMemo.walker_to_mbar_indices(walker_memo, x)


def test_MBARResult():
    energy = np.asarray([
        [0.2, 0.5, 0.7, 1.0, -2.3],
        [30.5, 42.5, 880.2, 30.1, 33.6]
    ])
    num_conf = np.asarray([3, 2])
    lambdas = np.asarray([0., 1])

    for parallel in [False, True]:
        obj = MBARResult(lambdas, energy, num_conf, parallel=parallel)
        assert np.isclose(obj.computeFreeEnergies(lambdas)[-1], 33.64829833917194)
        assert np.isclose(obj.computeFreeEnergies(lambdas[::-1])[0], 33.64829833917194)
        assert np.isclose(obj.computeFreeEnergies([lambdas[0]])[-1], 0)
        assert np.isclose(obj.computeFreeEnergies([lambdas[1]])[-1], 33.64829833917194)

        energy_pert = np.asarray([[100., 30., 80., 20., 10.]])
        free_energy_pert = obj.computeFreeEnergies([0.5], energy=energy_pert)
        assert np.isclose(free_energy_pert, 13.466387389648583)
        observable = np.asarray([[520., 518., 388., 100., 80.]])
        assert np.isclose(obj.computeExpectations(observable, energy=energy_pert), 80.00042764083992, rtol=1e-8)
        assert np.isclose(obj.computeExpectations(observable, energy=energy_pert, free_energy=free_energy_pert),
                          80.00042764083992, rtol=1e-8)

        energy_full = [energy[0, :], energy_pert[0, :], energy[1, :]]
        lambdas_full = [0, 0.5, 1]
        num_conf_full = [3, 0, 2]
        obj = MBARResult(lambdas_full, energy_full, num_conf_full, parallel=parallel)
        expected_free_energy = [0, 13.466387389648583, 33.64829833917194]
        assert np.allclose(obj.free_energy, expected_free_energy)


def test_MBARResult_multiple():
    energy = np.asarray([[
        [0.2, 0.5, 0.7, 1.0, -2.3],
        [30.5, 42.5, 880.2, 30.1, 33.6]
    ]])
    num_conf = np.asarray([[3, 2]])
    lambdas = np.asarray([0., 1])

    obj = MBARResult(lambdas, energy, num_conf)
    assert np.isclose(obj.computeFreeEnergies(lambdas)[0, -1], 33.64829833917194)
    assert np.isclose(obj.computeFreeEnergies(lambdas[::-1])[0, 0], 33.64829833917194)
    assert np.isclose(obj.computeFreeEnergies([lambdas[0]])[0, -1], 0)
    assert np.isclose(obj.computeFreeEnergies([lambdas[1]])[0, -1], 33.64829833917194)

    energy_pert = np.asarray([[[100., 30., 80., 20., 10.]]])
    free_energy_pert = obj.computeFreeEnergies([0.5], energy=energy_pert)
    assert np.isclose(free_energy_pert[0], 13.466387389648583)
    observable = np.asarray([[[520., 518., 388., 100., 80.]]])
    assert np.isclose(obj.computeExpectations(observable, energy=energy_pert), 80.00042764083992, rtol=1e-8)
    assert np.isclose(obj.computeExpectations(observable, energy=energy_pert, free_energy=free_energy_pert),
                      80.00042764083992, rtol=1e-8)

def test_MBARResult_jagged():
    energy = [[
        [0.2, 0.5, 0.7, 1.0, -2.3],
        [30.5, 42.5, 880.2, 30.1, 33.6]
    ], [
        [0.5, 0.7, -2.3, -2.3],
        [42.5, 880.2, 33.6, 33.6]
    ]]
    num_conf = [[3, 2], [2, 2]]
    lambdas = np.asarray([0., 1])

    obj = MBARResult(lambdas, energy, num_conf)
    expected_free_energy = np.asarray([[0., 33.64829833917194], [0., 39.313316716289265]])
    assert np.allclose(obj.computeFreeEnergies(lambdas), expected_free_energy)
    assert np.allclose(obj.computeFreeEnergies(lambdas[::-1]), expected_free_energy[:, ::-1])
    assert np.allclose(obj.computeFreeEnergies([lambdas[0]]), expected_free_energy[:, [0]])
    assert np.allclose(obj.computeFreeEnergies([lambdas[1]]), expected_free_energy[:, [1]])

    energy_pert = [[
        [100., 30., 80., 20., 10.],
        [30.5, 42.5, 880.2, 30.1, 33.6],
        [0.2, 0.5, 0.7, 1.0, -2.3]
    ], [
        [30., 80., 10., 10.],
        [42.5, 880.2, 33.6, 33.6],
        [0.5, 0.7, -2.3, -2.3]
    ]]
    free_energy_pert = obj.computeFreeEnergies([0.5, 1, 0], energy=energy_pert)
    expected_free_energy_pert = np.asarray([[13.466387389648583, 33.64829833917194, 0.],
                                    [15.745717379101794, 39.313316716289265, 0.]])
    assert np.allclose(free_energy_pert, expected_free_energy_pert)
    energy_pert = [[[100., 30., 80., 20., 10.]], [[30., 80., 10., 10.]]]
    observable = [[[520., 518., 388., 100., 80.]], [[518., 388., 80., 80.]]]
    free_energy_pert = expected_free_energy_pert[:, [0]]
    expected_result = np.asarray([[80.00042764083992], [80.00021798194851]])
    assert np.allclose(obj.computeExpectations(observable, energy=energy_pert), expected_result, rtol=1e-8)
    assert np.allclose(obj.computeExpectations(observable, energy=energy_pert, free_energy=free_energy_pert),
                       expected_result, rtol=1e-8)

def test_BAR_MBAR():
    # test correctness of regular FE calculation
    for cls in [BAR, MBAR]:
        obj = cls(walker_memo)
        assert np.isclose(obj(0, 1), 33.64829833917194)
    
        # test correctness of bootstrapped FE calculation
        indices = [2, 4, 4, 3]
        obj_scrambled = cls(walker_memo)
        assert np.isclose(obj_scrambled(0, 1, custom_indices=indices), 39.313316716289265)
    
        # test correctness of the memo
        assert np.isclose(obj_scrambled(1, 0), -39.313316716289265)
        assert np.isclose(obj_scrambled(0, 1), 39.313316716289265)
        model = obj_scrambled.getModel(min_lambda=0, max_lambda=1)[0]
        model._lambda_memo = np.asarray([0, 0.2, 1., 2.])
        model._free_energy_memo = np.asarray([0, 35., 39.313316716289265, 45.])
        assert np.isclose(obj_scrambled(1, 0), -39.313316716289265)
        assert np.isclose(obj_scrambled(0, 1), 39.313316716289265)

    # test correctness of the expectation calculations
    u_pert = np.asarray([[100., 20., 30., 80., 10.]])
    observable = np.asarray([[520., 100., 518., 388., 80.]])
    assert np.isclose(obj.computeExpectations(observable, energy=u_pert), 80.00042764083992, rtol=1e-8)


def test_Ensemble():
    for parallel in [False, True]:
        for cls in [EnsembleBAR, EnsembleMBAR]:
            obj = cls(walker_memo, parallel=parallel, n_bootstraps=10)
            obj._generate_custom_indices = lambda *args: [[0, 1, 2, 3, 4], [4, 2, 4, 3]]

            assert np.allclose(obj(0, 1), [33.64829833917194, 39.313316716289265])
            assert np.allclose(obj(1, 0), [-33.64829833917194, -39.313316716289265])

            u_pert = np.asarray([[100., 20., 30., 80., 10.]])
            A_n = np.asarray([[520., 100., 518., 388., 80.]])
            assert np.allclose(obj.computeExpectations(A_n, energy=u_pert), [[80.00042764083992], [80.00021798194851]],
                               rtol=1e-8)

        expected_free_energy = [13.466387389648583, 15.745717379101794]
        assert np.allclose(obj(0, 0.5, energy=[[0.2, 1.0, 0.5, 0.7, -2.3], u_pert[0]]), expected_free_energy)