from numpy import sign as _sign


class GreedyBisectingMinimiser:
    @classmethod
    def minimise(cls, func, threshold_y, pivot_x, maximum_x, pivot_y=None, minimum_x=None, maxfun=100, tol=1e-8,
                 *args, **kwargs):
        x_0 = pivot_x
        y_0 = pivot_y if pivot_y is not None else func(x_0, *args, **kwargs)
        x_1 = maximum_x
        y_1 = func(x_1, *args, **kwargs)
        minimum_x = pivot_x if minimum_x is None else minimum_x

        if y_0 == threshold_y:
            return x_0

        sgn = _sign(y_0 - threshold_y)
        if _sign(y_1 - threshold_y) == sgn:
            return x_1

        x_within_threshold = [x_0, minimum_x]

        for i in range(maxfun):
            x_curr = max(0.5 * (x_0 + x_1), minimum_x)
            y_curr = func(x_curr, *args, **kwargs)

            if abs(y_curr - threshold_y) <= abs(tol * threshold_y):
                return x_curr

            if _sign(y_curr - threshold_y) == sgn:
                x_0 = x_curr
                x_within_threshold += [x_curr]
            else:
                if x_curr <= minimum_x:
                    return minimum_x
                x_1 = x_curr

        return max(x_within_threshold)
