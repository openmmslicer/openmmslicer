import numpy as np

from slicer.mbar import MBARResult


def test_MBARResult():
    u_kn = np.asarray([
        [0.2, 0.5, 0.7, 1.0, -2.3],
        [30.5, 42.5, 880.2, 30.1, 33.6]
    ])
    N_k = np.asarray([3, 2])

    # test correctness of regular FE calculation
    mbar = MBARResult(u_kn, N_k)
    assert np.isclose(mbar.getFreeEnergyDifferences()[-1], 33.64829833917194)

    # test that the FE calculation memo works
    mbar._mbar = None
    assert np.isclose(mbar.getFreeEnergyDifferences()[-1], 33.64829833917194)

    # test correctness of the bootstrapped FE calculation
    indices = [1, 4, 4, 2]
    mbar = MBARResult(u_kn, N_k, bootstrapped_indices=indices)
    mbar_scrambled = MBARResult(u_kn, N_k, bootstrapped_indices=[4, 4, 1, 2])
    assert np.isclose(mbar.getFreeEnergyDifferences()[-1], 39.313316716289265)
    assert np.sum(mbar.getFreeEnergyDifferences() == mbar_scrambled.getFreeEnergyDifferences()) == 2

    # test correctness of the perturbed bootstrapped FE calculation
    assert np.isclose(mbar.computePerturbedFreeEnergies(u_kn, memo_key="a")[-1], 39.313316716289265)
    assert np.sum(mbar.computePerturbedFreeEnergies(u_kn) == mbar_scrambled.computePerturbedFreeEnergies(u_kn)) == 2

    # test that the perturbed FE memo works
    assert np.isclose(mbar.computePerturbedFreeEnergies(memo_key="a")[-1], 39.313316716289265)

    # test correctness of the expectation calculations
    u_pert = np.asarray([[100., 30., 80., 20., 10.]])
    A_n = np.asarray([[520., 518., 388., 100., 80.]])
    assert np.isclose(mbar.computeExpectations(A_n, u_pert, memo_key="b"), 80.00021798)
    assert np.sum(mbar.computeExpectations(A_n, u_pert) == mbar_scrambled.computeExpectations(A_n, u_pert)) == 1

    # test that the expectation memo works
    assert np.isclose(mbar.computeExpectations(memo_key="b"), 80.00021798)
