from functools import wraps


def run_check(f):
    @wraps(f)
    def wrapper(self, *args, **kwags):
        if not self._fitted:
            raise ValueError("Model not yet fit to data")
        return f(self, *args, **kwags)

    return wrapper
