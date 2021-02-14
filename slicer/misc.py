import functools as _functools
import inspect as _inspect


class LoggerWriter:
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
    def decorated_func(func):
        @_functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            frames = _inspect.getouterframes(_inspect.currentframe())[1:]
            recurse = any(x for x in frames if x.function == func.__name__ and "self" in x.frame.f_locals.keys() and
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
