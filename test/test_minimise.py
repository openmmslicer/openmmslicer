from slicer.minimise import BisectingMinimiser


def test_minimise():
    func = lambda x: 10 * x ** 2
    assert BisectingMinimiser.minimise(func, 10, 0.1, 1000, initial_guess_x=None, maxfun=1000, tol=0) == 1
    assert BisectingMinimiser.minimise(func, 10, 0.1, 1000, initial_guess_x=0.1001, maxfun=1000, tol=0) == 1
    assert BisectingMinimiser.minimise(func, 10, 0.1, 1000, initial_guess_x=10, maxfun=1000, tol=0) == 1
    assert BisectingMinimiser.minimise(func, 10, 0.1, 1.1, initial_guess_x=None, maxfun=1000, tol=0) == 1
    assert BisectingMinimiser.minimise(func, 10, 0.1, 1.1, initial_guess_x=1.2, maxfun=1000, tol=0) == 1
    assert BisectingMinimiser.minimise(func, 10, 0.1, 1.1, initial_guess_x=1.3, minimum_x=1.2, maxfun=1000, tol=0) == 1.1
    assert BisectingMinimiser.minimise(func, 10, 0.1, 1.1, minimum_x=1.2, maxfun=1000, tol=0) == 1.1

    func = lambda x: -10 * x ** 2
    assert BisectingMinimiser.minimise(func, -10, -0.1, -1000, initial_guess_x=None, maxfun=1000, tol=0) == -1
    assert BisectingMinimiser.minimise(func, -10, -0.1, -1000, initial_guess_x=-0.1001, maxfun=1000, tol=0) == -1
    assert BisectingMinimiser.minimise(func, -10, -0.1, -1000, initial_guess_x=-10, maxfun=1000, tol=0) == -1
    assert BisectingMinimiser.minimise(func, -10, -0.1, -1.1, initial_guess_x=None, maxfun=1000, tol=0) == -1
    assert BisectingMinimiser.minimise(func, -10, -0.1, -1.1, initial_guess_x=-1.2, maxfun=1000, tol=0) == -1
    assert BisectingMinimiser.minimise(func, -10, -0.1, -1.1, initial_guess_x=-1.3, minimum_x=-1.2, maxfun=1000, tol=0) == -1.1
    assert BisectingMinimiser.minimise(func, -10, -0.1, -1.1, minimum_x=-1.2, maxfun=1000, tol=0) == -1.1
