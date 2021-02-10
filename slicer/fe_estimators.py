import abc as _abc
from collections import defaultdict as _defaultdict
import logging as _logging

from cached_property import cached_property as _cached_property
import numba as _nb
import numpy as _np
from scipy.interpolate import interp1d as _interp1d
from scipy.optimize import minimize as _minimize

_logger = _logging.getLogger(__name__)

__all__ = ["AbstractFEEstimator", "MBAR", "BAR", "EnsembleMBAR", "EnsembleBAR"]


class ParallelNumbaFunction:
    def __init__(self, serial_func=None, parallel_func=None):
        self.serial_func = _nb.njit(fastmath=True, parallel=False, cache=True)(serial_func)
        self.parallel_func = _nb.njit(fastmath=True, parallel=True, cache=True)(parallel_func)

    def __call__(self, *args, parallel=False, **kwargs):
        func = self.parallel_func if parallel else self.serial_func
        return func(*args, **kwargs)


def _compute_log_weights(bias_energy, energy):
    res = _np.empty(energy.shape[1])
    for j in _nb.prange(energy.shape[1]):
        exp_biased_energy_j = _np.empty(energy.shape[0])
        exp_biased_energy_j[0] = bias_energy[0] + energy[0][j]
        min_energy_j = exp_biased_energy_j[0]
        sum_exp_biased_energy_j = 0.

        # first loop adds bias and calculates minima
        for i in _nb.prange(1, energy.shape[0]):
            exp_biased_energy_j[i] = bias_energy[i] + energy[i][j]
            if exp_biased_energy_j[i] < min_energy_j:
                min_energy_j = exp_biased_energy_j[i]

        # second loop uses the calculated minima, exponentiates and stores into a sum
        for i in _nb.prange(energy.shape[0]):
            sum_exp_biased_energy_j += _np.exp(-exp_biased_energy_j[i] + min_energy_j)
        res[j] = -_np.log(sum_exp_biased_energy_j) + min_energy_j
    return res


_compute_log_weights_numba = ParallelNumbaFunction(_compute_log_weights, _compute_log_weights)


def _compute_free_energies(energy, log_weights):
    res = _np.zeros(energy.shape[0])
    for i in _nb.prange(energy.shape[0]):
        weighted_energy_i = _np.empty(energy.shape[1])
        weighted_energy_i[0] = -energy[i][0] + log_weights[0]
        max_energy_i = weighted_energy_i[0]

        # first loop adds bias and calculates minima
        for j in _nb.prange(1, energy.shape[1]):
            weighted_energy_i[j] = -energy[i][j] + log_weights[j]
            if weighted_energy_i[j] > max_energy_i:
                max_energy_i = weighted_energy_i[j]

        # second loop uses the calculated minima, exponentiates and calculates the free energies
        for j in _nb.prange(energy.shape[1]):
            res[i] += _np.exp(weighted_energy_i[j] - max_energy_i)
        res[i] = -_np.log(res[i]) - max_energy_i
    return res


_compute_free_energies_numba = ParallelNumbaFunction(_compute_free_energies, _compute_free_energies)


def _normalised_expectations(observable, energy, log_weights):
    res = _np.zeros(energy.shape[0])
    logp = -energy + log_weights
    for i in _nb.prange(energy.shape[0]):
        # first loop creates unnormalised logp and calculates maxima
        logp_i = _np.empty(energy.shape[1])
        logp_i[0] = -energy[i][0] + log_weights[0]
        max_logp_i = logp_i[0]
        for j in _nb.prange(1, energy.shape[1]):
            logp_i[j] = -energy[i][j] + log_weights[j]
            if logp_i[j] > max_logp_i:
                max_logp_i = logp_i[j]

        # second loop uses the calculated maxima, exponentiates and accumulates the average and the norm
        norm_factor = 0.
        for j in _nb.prange(energy.shape[1]):
            p_ij = _np.exp(logp[i][j] - max_logp_i)
            res[i] += observable[i][j] * p_ij
            norm_factor += p_ij
        res[i] /= norm_factor
    return res


_normalised_expectations_numba = ParallelNumbaFunction(_normalised_expectations, _normalised_expectations)


def _unnormalised_expectations(observable, energy, log_weights, free_energy):
    res = _np.zeros(energy.shape[0])
    for i in _nb.prange(len(res)):
        for j in _nb.prange(len(log_weights)):
            res[i] += observable[i][j] * _np.exp(log_weights[j] - energy[i][j] + free_energy[i])
    return res


_unnormalised_expectations_numba = ParallelNumbaFunction(_unnormalised_expectations, _unnormalised_expectations)


def _expectations_numba(observable, energy, log_weights, free_energy=None, parallel=False):
    if free_energy is None:
        return _normalised_expectations_numba(observable, energy, log_weights, parallel=parallel)
    else:
        return _unnormalised_expectations_numba(observable, energy, log_weights, free_energy, parallel=parallel)


def _MBAR_loss_and_grad_numba_serial(bias_energy, energy, num_conf_ratio):
    # based on the Fast MBAR method: https://doi.org/10.1021/acs.jctc.8b01010
    grad = _np.zeros(energy.shape[0])
    loss = 0.

    for j in range(energy.shape[1]):
        exp_biased_energy_j = _np.empty(energy.shape[0])
        exp_biased_energy_j[0] = bias_energy[0] + energy[0][j]
        min_energy_j = exp_biased_energy_j[0]
        sum_exp_biased_energy_j = 0.

        # first loop adds bias and calculates minima
        for i in range(1, energy.shape[0]):
            exp_biased_energy_j[i] = bias_energy[i] + energy[i][j]
            if exp_biased_energy_j[i] < min_energy_j:
                min_energy_j = exp_biased_energy_j[i]

        # second loop uses the calculated minima, exponentiates and stores into a sum
        for i in range(energy.shape[0]):
            exp_biased_energy_j[i] = _np.exp(-exp_biased_energy_j[i] + min_energy_j)
            sum_exp_biased_energy_j += exp_biased_energy_j[i]

        # third loop updates the gradient (and later the loss) based on the calculated exponentials and sums
        for i in range(energy.shape[0]):
            grad[i] -= exp_biased_energy_j[i] / sum_exp_biased_energy_j
        loss += _np.log(sum_exp_biased_energy_j) - min_energy_j

    # the final loop normalises and adds the final terms, which are dependent on num_conf_ratio
    loss /= energy.shape[1]
    for i in range(energy.shape[0]):
        grad[i] = grad[i] / energy.shape[1] + num_conf_ratio[i]
        loss += num_conf_ratio[i] * bias_energy[i]

    return loss, grad


def _MBAR_loss_and_grad_numba_parallel(bias_energy, energy, num_conf_ratio):
    # based on the Fast MBAR method: https://doi.org/10.1021/acs.jctc.8b01010
    # this is a slightly slower implementation due to a bug in Numba (https://github.com/numba/numba/issues/3681)
    # TODO: refactor when the Numba issue is fixed
    grad = _np.zeros(energy.shape[0])
    thread_safe_grad = grad[:]
    loss = 0.

    for j in _nb.prange(energy.shape[1]):
        exp_biased_energy_j = _np.empty(energy.shape[0])
        exp_biased_energy_j[0] = bias_energy[0] + energy[0][j]
        min_energy_j = exp_biased_energy_j[0]
        sum_exp_biased_energy_j = 0.

        # first loop adds bias and calculates minima
        for i in _nb.prange(1, energy.shape[0]):
            exp_biased_energy_j[i] = bias_energy[i] + energy[i][j]
            if exp_biased_energy_j[i] < min_energy_j:
                min_energy_j = exp_biased_energy_j[i]

        # second loop uses the calculated minima, exponentiates and stores into a sum
        for i in _nb.prange(energy.shape[0]):
            exp_biased_energy_j[i] = _np.exp(-exp_biased_energy_j[i] + min_energy_j)
            sum_exp_biased_energy_j += exp_biased_energy_j[i]

        # no loop here, which results in a slight decline in performance, but it is needed to ensure correct results
        thread_safe_grad -= exp_biased_energy_j / sum_exp_biased_energy_j
        loss += _np.log(sum_exp_biased_energy_j) - min_energy_j

    # the final loop normalises and adds the final terms, which are dependent on num_conf_ratio
    loss /= energy.shape[1]
    grad = grad / energy.shape[1] + num_conf_ratio
    for i in _nb.prange(energy.shape[0]):
        loss += num_conf_ratio[i] * bias_energy[i]

    return loss, grad


_MBAR_loss_and_grad_numba = ParallelNumbaFunction(_MBAR_loss_and_grad_numba_serial, _MBAR_loss_and_grad_numba_parallel)


def _solve_MBAR(energy, num_conf, initial_free_energy=None, parallel=False, method='L-BFGS-B', tol=1e-8, **kwargs):
    # prepare for the optimisation
    optfunc = _MBAR_loss_and_grad_numba_parallel if parallel else _MBAR_loss_and_grad_numba
    energy = _np.asarray(energy)
    num_conf_ratio = _np.round(num_conf.astype(dtype=_np.float64), 0)
    zero_indices = (num_conf_ratio == 0)
    nonzero_indices = ~zero_indices
    if _np.sum(zero_indices) == 0:
        energy_nonzero = energy
        num_conf_ratio_nonzero = num_conf_ratio
    else:
        energy_nonzero = energy[nonzero_indices, :]
        num_conf_ratio_nonzero = num_conf_ratio[nonzero_indices]
    num_conf_ratio_nonzero /= _np.sum(num_conf_ratio)
    if initial_free_energy is None:
        initial_free_energy = _np.zeros(num_conf_ratio_nonzero.size)
    else:
        initial_free_energy = _np.asarray(initial_free_energy)[nonzero_indices]
    initial_bias_energy = -initial_free_energy - _np.log(num_conf_ratio_nonzero)

    # optimise
    result = _minimize(optfunc, initial_bias_energy, args=(energy_nonzero, num_conf_ratio_nonzero), jac=True,
                       method=method, tol=tol, options=kwargs)

    # calculate observables of interest and return
    log_weights = _compute_log_weights_numba(result['x'], energy_nonzero, parallel=parallel)
    free_energy = _compute_free_energies_numba(energy, log_weights, parallel=parallel)
    log_weights += free_energy[0]
    free_energy -= free_energy[0]

    return free_energy, log_weights, result


class MBARResult:
    def __init__(self, lambdas, energy, num_conf, parallel=True, **kwargs):
        self._energy = self._cast(energy, 2)
        self._num_conf = self._cast(num_conf, 1)
        self._lambda_memo = _np.asarray(lambdas)
        self._validate(self._energy, num_conf=self._num_conf, lambdas=self._lambda_memo)
        self.parallel = parallel

        if isinstance(self._energy, _np.ndarray) and len(self._energy.shape) == 2:
            self._free_energy, self._log_weights, _ = _solve_MBAR(self._energy, self._num_conf, parallel=self.parallel,
                                                                  **kwargs)
            self._single_model = True
        else:
            results = [_solve_MBAR(self._energy[i], self._num_conf[i], parallel=self.parallel, **kwargs)
                       for i in range(len(self._energy))]
            self._free_energy = self._cast([x[0] for x in results], 1)
            self._log_weights = self._cast([x[1] for x in results], 1)
            self._single_model = False

        self._free_energy_memo = self.free_energy

    @staticmethod
    def _cast(arr, expected_dimensions):
        try:
            arr_return = _np.asarray(arr)
            if arr_return.dtype == 'object':
                raise ValueError
        except ValueError:
            arr_return = [_np.asarray(x) for x in arr]
            if any(len(x.shape) != expected_dimensions for x in arr_return):
                raise ValueError("Need to pass a 2D or a 3D array or a nested list of 2D arrays")
        return arr_return

    @staticmethod
    def _validate(energy, lambdas=None, num_conf=None, free_energy=None, log_weights=None, observable=None):
        def validate_2D(energy, lambdas=None, num_conf=None, free_energy=None, log_weights=None, observable=None):
            n_lambdas, n_samples = energy.shape
            expected_shapes = {
                "lambdas": (n_lambdas,),
                "num_conf": (n_lambdas,),
                "free_energy": (n_lambdas,),
                "log_weights": (n_samples,),
                "observable": (n_lambdas, n_samples),
            }
            for key in expected_shapes.keys():
                if locals()[key] is not None and locals()[key].shape != expected_shapes[key]:
                    raise ValueError(f"Shape mismatch for {key}: expected {expected_shapes[key]}, received "
                                     f"{locals()[key].shape}")
            if num_conf is not None and _np.sum(num_conf) != n_samples:
                raise ValueError("The sum of num_conf must be equal to the number of samples")

        if isinstance(energy, _np.ndarray) and len(energy.shape) == 2:
            validate_2D(energy, lambdas=lambdas, num_conf=num_conf, free_energy=free_energy, log_weights=log_weights,
                        observable=observable)
        else:
            n_models = len(energy)
            f_locals = locals()
            variables_to_check = [x for x in ["lambdas", "num_conf", "free_energy", "log_weights", "observable"]
                                  if f_locals[x] is not None]
            if not all(len(f_locals[x]) == n_models for x in variables_to_check if x != "lambdas"):
                raise ValueError("All variables must have the same length")
            for i in range(n_models):
                kwargs = {x: (f_locals[x][i] if x != "lambdas" else f_locals[x]) for x in variables_to_check}
                validate_2D(energy[i], **kwargs)

    @_cached_property
    def free_energy(self):
        return self._free_energy

    @property
    def log_weights(self):
        return self._log_weights

    def computeFreeEnergies(self, lambdas, energy=None):
        # pre-sort the lambdas for convenience
        argsort = _np.argsort(lambdas)
        lambdas = _np.asarray(lambdas)[argsort]

        # retrieve the cached free energies and calculate the uncached ones
        if self._single_model:
            fes = _np.zeros(lambdas.size)
        else:
            fes = _np.zeros((len(self.log_weights), lambdas.size))
        fes[..., _np.isin(lambdas, self._lambda_memo)] = self._free_energy_memo[..., _np.isin(self._lambda_memo, lambdas)]
        indices_uncached = _np.isin(lambdas, self._lambda_memo, invert=True)
        lambdas_uncached = lambdas[indices_uncached]
        if lambdas_uncached.size:
            if energy is None:
                raise ValueError("Need to pass energy in order to calculate uncached free energies")
            else:
                energy = self._cast(energy, 2)
                self._validate(energy, lambdas=lambdas)

            if self._single_model:
                fes_uncached = _compute_free_energies_numba(energy[argsort[indices_uncached], :], self.log_weights,
                                                            parallel=self.parallel)
            else:
                fes_uncached = [_compute_free_energies_numba(energy[i][argsort[indices_uncached], :],
                                                             self._log_weights[i], parallel=self.parallel)
                                for i in range(len(self.log_weights))]
                fes_uncached = self._cast(fes_uncached, 1)

            fes[..., indices_uncached] = fes_uncached

            # update the cache with the uncached free energies
            lambda_memo_new = _np.concatenate([self._lambda_memo, lambdas_uncached])
            free_energy_memo_new = _np.concatenate([self._free_energy_memo, fes_uncached], axis=-1)
            argsort_cache = _np.argsort(lambda_memo_new)
            self._lambda_memo = lambda_memo_new[argsort_cache]
            self._free_energy_memo = free_energy_memo_new[..., argsort_cache]

        # restore the original order of the free energies and return
        fes = fes[..., _np.argsort(argsort)]
        return fes

    def computeExpectations(self, observable, energy=None, free_energy=None):
        if energy is None:
            energy = self._energy
            if free_energy is None:
                free_energy = self.free_energy
        observable = self._cast(observable, 2)
        energy = self._cast(energy, 2)
        if free_energy is not None:
            free_energy = self._cast(free_energy, 1)
        self._validate(energy, free_energy=free_energy, observable=observable)

        # it is crucial that the below steps are executed as fast as possible
        if free_energy is None:
            func, args = _normalised_expectations_numba, (observable, energy, self.log_weights)
        else:
            func, args = _unnormalised_expectations_numba, (observable, energy, self.log_weights, free_energy)

        if self._single_model:
            return _np.asarray(func(*args, parallel=self.parallel))
        else:
            return _np.asarray([func(*arg, parallel=self.parallel) for arg in zip(*args)])


class AbstractFEEstimator(_abc.ABC):
    def __init__(self, walker_memo, update_func=lambda self: self.walker_memo.timesteps + 1):
        self.walker_memo = walker_memo
        self.update_func = update_func

    @property
    def effective_sample_size(self):
        return self.walker_memo.timesteps

    @_abc.abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    @_abc.abstractmethod
    def computeFreeEnergies(self, *args, **kwargs):
        pass

    @_abc.abstractmethod
    def computeExpectations(self, *args, **kwargs):
        pass


class MBAR(AbstractFEEstimator):
    def __init__(self, *args, parallel=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.parallel = parallel

    @property
    def walker_memo(self):
        return self._walker_memo

    @walker_memo.setter
    def walker_memo(self, val):
        self._walker_memo = val
        self.reset()

    def reset(self):
        self._model_memo = _defaultdict(lambda: (0, None, _np.asarray([])))

    def _convert_indices(self, custom_indices=None, min_lambda=0., max_lambda=1., strict_lambdas=False):
        # generate walker indices from time indices, if applicable
        if custom_indices is None:
            custom_indices = _np.arange(len(self.walker_memo.relevant_walkers))
        else:
            custom_indices = _np.asarray(custom_indices)
        # constrain to the lambda range, if applicable
        if strict_lambdas or min_lambda != 0. or max_lambda != 1.:
            if strict_lambdas:
                valid_lambda_indices = _np.where((self.walker_memo.lambdas == max_lambda) |
                                                 (self.walker_memo.lambdas == min_lambda))[0]
            else:
                valid_lambda_indices = _np.where((self.walker_memo.lambdas <= max_lambda) &
                                                 (self.walker_memo.lambdas >= min_lambda))[0]
            custom_indices = custom_indices[_np.isin(custom_indices, valid_lambda_indices)]
        # convert walker indices to mbar indices
        custom_indices = self.walker_memo.walker_to_mbar_indices(custom_indices)
        return custom_indices

    def fit(self, custom_indices=None, min_lambda=0., max_lambda=1., strict_lambdas=False, old_model=None, **kwargs):
        # filter and convert the custom_indices
        custom_indices = self._convert_indices(custom_indices=custom_indices, min_lambda=min_lambda,
                                               max_lambda=max_lambda, strict_lambdas=strict_lambdas)

        if not custom_indices.size:
            model = None
        else:
            # create the MBAR arguments
            if old_model is not None:
                interp = _interp1d(old_model._lambda_memo, old_model._free_energy_memo)
                initial_free_energy = interp(self.walker_memo.unique_lambdas)
                kwargs["initial_free_energy"] = initial_free_energy - initial_free_energy[0]
            valid_lambdas, valid_num_conf = _np.unique(self.walker_memo.lambdas[custom_indices], return_counts=True)
            num_conf = _np.zeros(len(self.walker_memo.unique_lambdas))
            num_conf[_np.isin(self.walker_memo.unique_lambdas, valid_lambdas)] = valid_num_conf
            energy = self.walker_memo.energyMatrix(lambdas=self.walker_memo.unique_lambdas)[:, custom_indices]

            # run the MBAR model
            kwargs["parallel"] = self.parallel
            model = MBARResult(self.walker_memo.unique_lambdas, energy, num_conf, **kwargs)

        # update the memo
        next_calculation = self.walker_memo.timesteps + self.update_func(self)
        self._model_memo[(min_lambda, max_lambda)] = (next_calculation, model, custom_indices)

        return model, custom_indices

    def getModel(self, min_lambda=0., max_lambda=1., recalculate=False, custom_indices=None, **kwargs):
        next_calculation, model, old_custom_indices = self._model_memo[(min_lambda, max_lambda)]
        if self.walker_memo.timesteps >= next_calculation or recalculate:
            kwargs["old_model"] = model
            model, new_custom_indices = self.fit(min_lambda=min_lambda, max_lambda=max_lambda,
                                                 custom_indices=custom_indices, **kwargs)
            return model, new_custom_indices
        else:
            return model, old_custom_indices

    def __call__(self, lambda0, lambda1, **kwargs):
        fes = self.computeFreeEnergies([lambda0, lambda1], **kwargs)
        if fes is None:
            return _np.inf
        else:
            return fes[1] - fes[0]

    def computeFreeEnergies(self, lambdas, energy=None, **kwargs):
        model, custom_indices = self.getModel(**kwargs)
        if model is None:
            return None
        if energy is None:
            energy = self.walker_memo.energyMatrix(lambdas)
        energy = _np.asarray(energy)[:, custom_indices]
        return model.computeFreeEnergies(lambdas, energy=energy)

    def computeExpectations(self, observable, lambdas=None, energy=None, free_energy=None, **kwargs):
        kwargs.update(dict(min_lambda=0., max_lambda=1.))
        model, custom_indices = self.getModel(**kwargs)
        if lambdas is None and energy is None:
            raise ValueError("Need to supply either energy or lambdas")
        if energy is None:
            energy = self.walker_memo.energyMatrix(lambdas)
        observable = _np.asarray(observable)[:, custom_indices]
        energy = _np.asarray(energy)[:, custom_indices]
        return model.computeExpectations(observable, energy=energy, free_energy=free_energy)


def _MBAR_to_BAR(cls):
    class new_cls(cls):
        def fit(self, *args, **kwargs):
            kwargs["strict_lambdas"] = True
            return super().fit(*args, **kwargs)

        def computeFreeEnergies(self, lambdas, **kwargs):
            lambdas = _np.asarray(lambdas)
            if lambdas.size != 2:
                raise ValueError("BAR free energies can only be computed for two lambda values")
            kwargs.update(dict(min_lambda=_np.min(lambdas), max_lambda=_np.max(lambdas)))
            return super().computeFreeEnergies(lambdas, **kwargs)
    return new_cls


@_MBAR_to_BAR
class BAR(MBAR):
    pass


class EnsembleMBAR(MBAR):
    def __init__(self, *args, n_bootstraps=None, n_decorr=None, interval=None,
                 update_func=lambda self: 1 + 0.01 * self.effective_sample_size, **kwargs):
        super().__init__(*args, update_func=update_func, **kwargs)
        self.n_decorr = n_decorr
        self.n_bootstraps = n_bootstraps
        self.interval = interval

    def reset(self):
        self._model_memo = _defaultdict(lambda: (0, None, _np.asarray([[]])))

    @property
    def effective_sample_size(self):
        return self.walker_memo.timesteps // self.interval

    @property
    def interval(self):
        val = self._interval() if callable(self._interval) else self._interval
        if val is None or _np.isnan(val) or _np.isinf(val):
            return 1
        else:
            return max(1, int(round(float(val))))

    @interval.setter
    def interval(self, val):
        self._interval = val

    def _generate_custom_indices(self):
        # decorrelate, if applicable
        if self.interval != 1 and self.n_decorr is not None:
            n_decorr = max(1, min(self.n_decorr, self.interval))
            offsets = _np.random.choice(_np.arange(self.interval), size=n_decorr, replace=False)
            custom_indices = [_np.arange(self.walker_memo.timesteps - 1 - x, -1, -self.interval)[::-1] for x in offsets]
            custom_indices = [self.walker_memo.time_to_walker_indices(x) for x in custom_indices]
        else:
            custom_indices = [_np.arange(len(self.walker_memo.relevant_walkers))]
        final_indices = custom_indices

        # bootstrap, if applicable
        if self.n_bootstraps is not None:
            custom_indices = []
            for indices in final_indices:
                weights = self.walker_memo.weights[indices]
                size = _np.round(_np.sum(weights))
                weights /= _np.sum(weights)
                custom_indices += [_np.random.choice(indices, size=size, p=weights) for _ in range(self.n_bootstraps)]
        final_indices = custom_indices
        return final_indices

    def _convert_indices(self, *args, custom_indices=None, **kwargs):
        if custom_indices is None:
            custom_indices = [super(EnsembleMBAR, self)._convert_indices(*args, custom_indices=None, **kwargs)]
        else:
            custom_indices = [super(EnsembleMBAR, self)._convert_indices(*args, custom_indices=x, **kwargs)
                              for x in custom_indices]
        return [x for x in custom_indices if x.size]

    @staticmethod
    def _partition_from_indices(arr, indices):
        arr = _np.asarray(arr)
        if len(set(x.shape for x in indices)) == 1:
            arr_new = _np.empty((len(indices), *arr.shape))
        else:
            arr_new = [None] * len(indices)
        for i, batch in enumerate(indices):
            arr_new[i] = arr[..., _np.ravel(batch)]
        return arr_new

    def fit(self, custom_indices=None, min_lambda=0., max_lambda=1., strict_lambdas=False, old_model=None, **kwargs):
        # filter and convert the custom_indices
        custom_indices = self._convert_indices(custom_indices=custom_indices, min_lambda=min_lambda,
                                               max_lambda=max_lambda, strict_lambdas=strict_lambdas)

        if not len(custom_indices):
            model = None
        else:
            # create the MBAR arguments
            if old_model is not None:
                interp = _interp1d(old_model._lambda_memo, old_model._free_energy_memo[0])
                initial_free_energy = interp(self.walker_memo.unique_lambdas)
                kwargs["initial_free_energy"] = initial_free_energy - initial_free_energy[0]

            num_conf_all = _np.empty((len(custom_indices), len(self.walker_memo.unique_lambdas)))
            for i, batch in enumerate(custom_indices):
                valid_lambdas, valid_num_conf = _np.unique(self.walker_memo.lambdas[batch], return_counts=True)
                num_conf = _np.zeros(len(self.walker_memo.unique_lambdas))
                num_conf[_np.isin(self.walker_memo.unique_lambdas, valid_lambdas)] = valid_num_conf
                num_conf_all[i] = num_conf
            energy = self.walker_memo.energyMatrix(lambdas=self.walker_memo.unique_lambdas)
            energy_all = self._partition_from_indices(energy, custom_indices)

            # run the MBAR model
            kwargs["parallel"] = self.parallel
            model = MBARResult(self.walker_memo.unique_lambdas, energy_all, num_conf_all, **kwargs)

        # update the memo
        next_calculation = self.walker_memo.timesteps + self.update_func(self)
        self._model_memo[(min_lambda, max_lambda)] = (next_calculation, model, custom_indices)

        return model, custom_indices

    def __call__(self, lambda0, lambda1, **kwargs):
        fes = self.computeFreeEnergies([lambda0, lambda1], **kwargs)
        if fes is None:
            return _np.asarray([_np.inf])
        else:
            return _np.ravel(fes[:, 1] - fes[:, 0])

    def getModel(self, min_lambda=0., max_lambda=1., recalculate=False, **kwargs):
        next_calculation, model, old_custom_indices = self._model_memo[(min_lambda, max_lambda)]
        if self.walker_memo.timesteps >= next_calculation or recalculate:
            custom_indices = self._generate_custom_indices()
            kwargs.update(dict(old_model=model, custom_indices=custom_indices))
            model, new_custom_indices = self.fit(min_lambda=min_lambda, max_lambda=max_lambda, **kwargs)
            return model, new_custom_indices
        else:
            return model, old_custom_indices

    def computeFreeEnergies(self, lambdas, energy=None, **kwargs):
        model, custom_indices = self.getModel(**kwargs)
        if model is None:
            return None
        if energy is None:
            energy = self.walker_memo.energyMatrix(lambdas)
        energy_all = self._partition_from_indices(energy, custom_indices)
        return model.computeFreeEnergies(lambdas, energy=energy_all)

    def computeExpectations(self, observable, lambdas=None, energy=None, free_energy=None, **kwargs):
        kwargs.update(dict(min_lambda=0., max_lambda=1.))
        model, custom_indices = self.getModel(**kwargs)
        if lambdas is None and energy is None:
            raise ValueError("Need to supply either energy or lambdas")
        if energy is None:
            energy = self.walker_memo.energyMatrix(lambdas)
        observable_all = self._partition_from_indices(observable, custom_indices)
        energy_all = self._partition_from_indices(energy, custom_indices)
        return model.computeExpectations(observable_all, energy=energy_all, free_energy=free_energy)


@_MBAR_to_BAR
class EnsembleBAR(EnsembleMBAR):
    pass
