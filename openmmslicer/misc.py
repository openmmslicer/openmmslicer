import functools as _functools
import inspect as _inspect

import openmm.unit as _unit


class LoggerWriter:
    """A minimal writer compatible with the logging module."""
    def __init__(self, level):
        self.level = level
        self.closed = False

    def write(self, message):
        if not isinstance(message, str):
            message = message.decode()
        if message != '\n':
            self.level(message)

    def flush(self):
        pass


def norecurse(default_return_value=None):
    """Prevents the recursion of a specified function by returning a default value."""
    def decorated_func(func):
        @_functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            frames = _inspect.getouterframes(_inspect.currentframe())[1:]
            recurse = any(x for x in frames if x.function == func.__name__ and "self" in x.frame.f_locals and
                          x.frame.f_locals["self"] is self)
            if recurse:
                if callable(default_return_value):
                    return default_return_value(self)
                else:
                    return default_return_value
            else:
                return func(self, *args, **kwargs)
        return wrapper
    return decorated_func


def quantity_round(quantity, number, unit=_unit.nanosecond):
    """Rounds an openmm.Quantity expressed in particular units to a pre-supplied number of significant figures."""
    value = quantity.value_in_unit(unit)
    return round(value, number) * unit