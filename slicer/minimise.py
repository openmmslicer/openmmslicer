from numpy import sign as _sign


class GreedyBisectingMinimiser:
    """
    A bisection algorithm specifically tailored for maximising the next lambda value over a threshold.
    """
    @classmethod
    def minimise(cls, func, threshold_y, pivot_x, maximum_x, pivot_y=None, minimum_x=None, maxfun=100, tol=1e-8,
                 *args, **kwargs):
        """
        Finds the maximum lambda value within a threshold using bisection

        Parameters
        ----------
        func : function
            The function to be used for evaluation.
        threshold_y : float
            The threshold value of the function.
        pivot_x : float
            A pivot value of x which determines the desired side of the threshold.
        maximum_x : float
            The maximum lambda value.
        pivot_y : float
            The function evaluate at pivot_x. Default calls evaluation of func on pivot_x.
        minimum_x : float
            The minimum lambda value. Default is no minimum.
        maxfun : int
            The maximum number of calls to the function.
        tol : float
            The relative tolerance of the function.
        args
            Positional arguments to be passed to func.
        kwargs
            Keyword arguments to be passed to func.

        Returns
        -------
        x : float
            The maximum value of x within the threshold.
        """
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
