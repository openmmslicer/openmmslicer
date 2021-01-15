import numba as _nb
import numpy as _np


@_nb.guvectorize(
    [(_nb.float64[:], _nb.float64[:, :], _nb.float64[:, :], _nb.float64[:, :])],
    '(k),(m, n),(m, n)->(k, m)', nopython=True, target='cpu'
)
def interp_vectorised(x, xp, yp, res):
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
    def __init__(self, x, y, sort=True):
        lengths = _np.asarray([len(i) for i in x])
        unique_lengths = _np.unique(lengths)
        self.dimensional_indices = [(i, _np.where(lengths == i)[0]) for i in unique_lengths]
        self.n_samples = len(x)
        self.xs = [_np.asarray([x[i] for i in indices]) for _, indices in self.dimensional_indices]
        self.ys = [_np.asarray([y[i] for i in indices]) for _, indices in self.dimensional_indices]
        if sort:
            for x, y in zip(self.xs, self.ys):
                argsort = _np.argsort(x, axis=-1)
                x[:] = _np.take_along_axis(x, argsort, axis=-1)
                y[:] = _np.take_along_axis(y, argsort, axis=-1)

    def __call__(self, x_interp):
        x_interp = _np.asarray(x_interp)
        res = _np.empty((len(x_interp), self.n_samples))
        for (_, indices), x, y in zip(self.dimensional_indices, self.xs, self.ys):
            res[:, indices] = interp_vectorised(x_interp, x, y)
        return res
