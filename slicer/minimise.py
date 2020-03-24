from numpy import sign as _sign


class GreedyBisectingMinimiser:
    @classmethod
    def minimise(cls, func, threshold_y, minimum_x, maximum_x, minimum_y=None, maxfun=100, tol=1e-8, *args, **kwargs):
        x_0 = minimum_x
        y_0 = minimum_y if minimum_y is not None else func(x_0, *args, **kwargs)
        x_1 = maximum_x
        y_1 = func(x_1, *args, **kwargs)

        if y_0 == threshold_y:
            return x_0

        sgn = _sign(y_0 - threshold_y)
        if _sign(y_1 - threshold_y) == sgn:
            return x_1

        x_within_threshold = [x_0]

        for i in range(maxfun):
            x_curr = 0.5 * (x_0 + x_1)
            y_curr = func(x_curr, *args, **kwargs)

            if abs(y_curr - threshold_y) <= abs(tol * threshold_y):
                return x_curr

            if _sign(y_curr - threshold_y) == sgn:
                x_0 = x_curr
                x_within_threshold += [x_curr]
            else:
                x_1 = x_curr

        return max(x_within_threshold)
