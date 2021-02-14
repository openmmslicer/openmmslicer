import inspect as _inspect


def norecurse(default_return_value=None):
    def decorated_func(func):
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
