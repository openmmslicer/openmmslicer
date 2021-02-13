import numba as _nb
import numpy as _np

from slicer.fe_estimators import ParallelNumbaFunction as _ParallelNumbaFunction, EnsembleMBAR as _EnsembleMBAR


def _expected_transition_matrix(energy, free_energy, log_weights):
    # first loop generates the probability distributions
    log_probability = _np.empty(energy.shape)
    probability = _np.empty(energy.shape)
    for i in _nb.prange(energy.shape[0]):
        for j in _nb.prange(energy.shape[1]):
            # we store in probability for the expectations and in log_probability for the observables
            log_probability[i][j] = log_weights[j] - energy[i][j] + free_energy[i]
            probability[i][j] = _np.exp(log_probability[i][j])

    # second loop generates the transition matrix
    matrix_size = 2 * energy.shape[0] - 2
    transition_matrix = _np.zeros((matrix_size, matrix_size))
    for i in _nb.prange(energy.shape[0]):
        # calculate the forward probabilities, if applicable
        if i != energy.shape[0] - 1:
            transition_forward_i = _np.empty(energy.shape[1])
            for j in _nb.prange(energy.shape[1]):
                log_probability_diff = log_probability[i + 1][j] - log_probability[i][j]
                if log_probability_diff >= 0:
                    transition_forward_i[j] = 1.
                else:
                    transition_forward_i[j] = _np.exp(log_probability_diff)
                transition_matrix[2 * i, min(2 * i + 2, matrix_size - 1)] += transition_forward_i[j] * probability[i][j]
        # calculate the backward probabilities, if applicable
        if i != 0:
            transition_backward_i = _np.empty(energy.shape[1])
            for j in _nb.prange(energy.shape[1]):
                log_probability_diff = log_probability[i - 1][j] - log_probability[i][j]
                if log_probability_diff >= 0:
                    transition_backward_i[j] = 1.
                else:
                    transition_backward_i[j] = _np.exp(log_probability_diff)
                transition_matrix[2 * i - 1, max(2 * i - 3, 0)] += transition_backward_i[j] * probability[i][j]
        # calculate the direction swap probabilities, if applicable
        if i != 0 and i != energy.shape[0] - 1:
            for j in _nb.prange(energy.shape[1]):
                transition_swap_diff_ij = transition_backward_i[j] - transition_forward_i[j]
                if transition_swap_diff_ij > 0.:
                    transition_matrix[2 * i, 2 * i - 1] += transition_swap_diff_ij * probability[i][j]
                else:
                    transition_matrix[2 * i - 1, 2 * i] -= transition_swap_diff_ij * probability[i][j]

    # third loop normalises the transition matrix
    for i in _nb.prange(matrix_size):
        sum_i = 0.
        for j in _nb.prange(matrix_size):
            sum_i += transition_matrix[i][j]
        if sum_i < 1:
            transition_matrix[i][i] = 1. - sum_i
        elif _np.abs(sum_i - 1) > 1e-8:
            for j in _nb.prange(matrix_size):
                transition_matrix[i][j] /= sum_i

    return transition_matrix


_expected_transition_matrix_numba = _ParallelNumbaFunction(_expected_transition_matrix, _expected_transition_matrix)


class ExpectedRoundTripTime:
    def __init__(self, fe_estimator=None):
        self.fe_estimator = fe_estimator

    @staticmethod
    def _cast(lambdas):
        if len(lambdas) < 2:
            raise ValueError("Need a lambda array of at least two lambda values")
        lambdas = _np.ravel(lambdas)
        lambdas[lambdas < 0] = 0.
        lambdas[lambdas > 1] = 1.
        return lambdas

    def expectedTransitionMatrix(self, lambdas):
        lambdas = self._cast(lambdas)

        # obtain the relevant data from the estimator
        with self.fe_estimator.walker_memo.lock:
            energy = self.fe_estimator.walker_memo.energyMatrix(lambdas)
            free_energy = self.fe_estimator.computeFreeEnergies(lambdas, energy=energy)
            model, custom_indices = self.fe_estimator.getModel()
        log_weights = model.log_weights

        # call the vectorised expected transition matrix function
        parallel = self.fe_estimator.parallel
        if isinstance(self.fe_estimator, _EnsembleMBAR):
            energy = self.fe_estimator._partition_from_indices(energy, custom_indices)
            transition_matrices = [_expected_transition_matrix_numba(energy[i], free_energy[i], log_weights[i],
                                                                     parallel=parallel) for i in range(len(energy))]
            transition_matrix = _np.average(transition_matrices, axis=0)
        else:
            energy = energy[:, custom_indices]
            transition_matrix = _expected_transition_matrix_numba(energy, free_energy, log_weights, parallel=parallel)

        return transition_matrix

    def expectedTransitionTime(self, lambdas, lambda0=0., lambda1=1., target0=1, target1=0, costs=None,
                               transition_matrix=None):
        lambdas = self._cast(lambdas)
        idx0, idx1 = _np.where(lambdas == lambda0)[0][0], _np.where(lambdas == lambda1)[0][0]
        if idx0 == idx1:
            return 0.
        i = max(0, 2 * idx0 - 1) if target0 == 0 else min(2 * idx0, 2 * lambdas.shape[0] - 3)
        j = max(0, 2 * idx1 - 1) if target1 == 0 else min(2 * idx1, 2 * lambdas.shape[0] - 3)

        if transition_matrix is None:
            transition_matrix = self.expectedTransitionMatrix(lambdas)
        else:
            if not (transition_matrix.shape[0] == transition_matrix.shape[1] == 2 * (lambdas.size - 1)):
                raise ValueError("Invalid transition matrix shape supplied")
        transition_matrix_del = _np.delete(_np.delete(transition_matrix, i, axis=0), i, axis=1)
        if costs is None:
            costs = _np.ones(lambdas.shape)
        costs = _np.asarray([costs[0]] + [x for x in costs[1:-1] for _ in range(2)] + [costs[-1]])
        identity = _np.identity(transition_matrix_del.shape[0])
        b = _np.delete(transition_matrix @ costs, i, axis=0)
        j = j if j < i else j - 1

        try:
            tau = _np.linalg.solve(identity - transition_matrix_del, b)[j]
            if tau < 0:
                tau = _np.inf
        except _np.linalg.LinAlgError:
            tau = _np.inf

        return tau

    def expectedRoundTripTime(self, *args, lambda0=0., lambda1=1., **kwargs):
        lambda0, lambda1 = min(lambda0, lambda1), max(lambda0, lambda1)
        tau_fwd = self.expectedTransitionTime(*args, lambda0=lambda0, lambda1=lambda1, target0=1, target1=0, **kwargs)
        tau_bwd = self.expectedTransitionTime(*args, lambda0=lambda1, lambda1=lambda0, target0=0, target1=1, **kwargs)
        tau_total = tau_bwd + tau_fwd
        return tau_total
