import numba as _nb
import numpy as _np


@_nb.guvectorize(
    [(_nb.float64[:], _nb.float64[:, :], _nb.float64[:, :], _nb.float64[:, :])],
    '(k),(m, n),(m, n)->(k, m)', nopython=True, target='cpu', fastmath=True, cache=True
)
def _interp_vectorised(x, xp, yp, res):
    """A fast numba function which performs interpolation based on x values, y values and interpolated x values."""
    # TODO: generalise to extrapolation
    for i in _nb.prange(len(xp)):
        for j in _nb.prange(len(x)):
            # binary search which assumes xp is sorted and has unique values
            start = 0
            end = len(xp[i]) - 1
            while start <= end:
                mid = (start + end) // 2
                if x[j] == xp[i][mid]:
                    res[j][i] = yp[i][mid]
                    break
                elif x[j] == xp[i][mid + 1]:
                    res[j][i] = yp[i][mid + 1]
                    break
                elif xp[i][mid] < x[j] < xp[i][mid + 1]:
                    x0, x1 = xp[i][mid], xp[i][mid + 1]
                    y0, y1 = yp[i][mid], yp[i][mid + 1]
                    res[j][i] = y0 + (y1 - y0) * (x[j] - x0) / (x1 - x0)
                    break
                elif x[j] < xp[i][mid]:
                    end = mid - 1
                else:
                    start = mid + 1


class BatchLinearInterp:
    """
    A helper class used to linearly interpolate between two lists of x and y values with different lengths.

    Parameters
    ----------
    x : [list] or [np.ndarray]
        The list of lists of x values.
    y : [list] or [np.ndarray]
        The list of lists of y values.
    """
    def __init__(self, x, y, sort=True):
        min_len = min(len(x), len(y))
        x, y = x[:min_len], y[:min_len]
        lengths_x, lengths_y = _np.asarray([len(i) for i in x]), _np.asarray([len(i) for i in y])
        if not _np.array_equal(lengths_x, lengths_y):
            raise ValueError("Need x and y to be the same shape")
        unique_lengths = _np.unique(lengths_x)
        self._dimensional_indices = [(i, _np.where(lengths_x == i)[0]) for i in unique_lengths]
        self._n_samples = min_len
        self._xs = [_np.asarray([x[i] for i in indices]) for _, indices in self._dimensional_indices]
        self._ys = [_np.asarray([y[i] for i in indices]) for _, indices in self._dimensional_indices]
        if sort:
            for x, y in zip(self._xs, self._ys):
                argsort = _np.argsort(x, axis=-1)
                x[:] = _np.take_along_axis(x, argsort, axis=-1)
                y[:] = _np.take_along_axis(y, argsort, axis=-1)

    def __call__(self, x_interp):
        """
        Returns the interpolated values for a sequence of x values.

        Parameters
        ----------
        x_interp : list or np.ndarray
            The sequence of x values. Their length must be the same as the lists used to initialise the class.

        Returns
        -------
        y_interp : np.ndarray
            The interpolated y values.
        """
        x_interp = _np.asarray(x_interp)
        res = _np.empty((len(x_interp), self._n_samples))
        for (_, indices), x, y in zip(self._dimensional_indices, self._xs, self._ys):
            res[:, indices] = _interp_vectorised(x_interp, x, y)
        return res
