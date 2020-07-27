from numpy import sign as _sign


class BisectingMinimiser:
    """
    A bisection algorithm specifically tailored for minimising/maximising the next lambda value over a threshold.
    """
    @classmethod
    def minimise(self, func, threshold_y, current_x, desired_x, minimum_x=None, initial_guess_x=None, current_y=None,
                 maxfun=100, tol=1e-8, *args, **kwargs):
        """
        Finds the maximum lambda value within a threshold using bisection

        Parameters
        ----------
        func : function
            The function to be used for evaluation.
        threshold_y : float
            The threshold value of the function.
        current_x : float
            The current value of x which determines the desired side of the threshold.
        desired_x : float
            The target value of x.
        minimum_x : float, None
            The minimum allowed value of x. Default is no minimum.
        initial_guess_x : float, None
            An optional initial value to be supplied for the first iteration of the algorithm. Subsequent iterations
            will exponentially increase (initial_guess_x - current_x) until the region containing the threshold is found.
            Afterwards, regular bisection will yield the exact value of the threshold.
        current_y : float, None
            An optional value of y corresponding to current_x. This is used to evaluate the desired side of the threshold.
            If None, func is evaluated at current_x.
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
        if initial_guess_x is not None:
            assert not (initial_guess_x <= current_x <= desired_x) and not (initial_guess_x >= current_x >= desired_x)
        if minimum_x is not None:
            assert not (minimum_x <= current_x <= desired_x) and not (minimum_x >= current_x >= desired_x)

        x_0 = current_x
        y_0 = current_y if current_y is not None else func(x_0, *args, **kwargs)
        x_1 = desired_x
        y_1 = func(x_1, *args, **kwargs)

        # edge cases
        if y_0 == threshold_y:
            return x_0

        sgn = _sign(y_0 - threshold_y)
        if _sign(y_1 - threshold_y) == sgn:
            return x_1

        sortfunc = max if desired_x > current_x else min
        minimum_x = x_0 if minimum_x is None else -sortfunc(-minimum_x, -desired_x)
        x_within_threshold = [x_0, minimum_x]
        delta_x = None if initial_guess_x is None else initial_guess_x - current_x

        for i in range(maxfun):
            if delta_x is None:
                proposal = 0.5 * (x_0 + x_1)
            else:
                proposal = x_0 + delta_x
                delta_x *= 2 ** (i + 1)
            x_curr = sortfunc(minimum_x, proposal)
            y_curr = func(x_curr, *args, **kwargs)

            if abs(y_curr - threshold_y) <= abs(tol * threshold_y):
                return x_curr

            if _sign(y_curr - threshold_y) == sgn:
                x_0 = x_curr
                x_within_threshold += [x_curr]
            else:
                if sortfunc(x_curr, minimum_x) == minimum_x:
                    return minimum_x
                x_1 = x_curr
                delta_x = None

        return sortfunc(x_within_threshold)