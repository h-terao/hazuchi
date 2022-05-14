class Registry:
    """A registry class that allows any functions and classes by the registered keys."""

    def __init__(self):
        self._registry = {}

    def __len__(self):
        return len(self._registry)

    def items(self):
        return self._registry.items()

    def values(self):
        return self._registry.values()

    def keys(self):
        return self._registry.keys()

    def __dir__(self):
        return list(self.keys())

    def __iter__(self):
        return self._registry

    def __getitem__(self, key):
        return self.get(key)

    def get(self, key, *default):
        return self._registry.get(key, *default)

    def register(self, func_or_class=None, *, key: str):
        """Register function or class into the registry.

        Usage:
            >>> registry = Registry()
            >>> registry.register(func, key=func_name)

            >>> @registry.register(key=func_name)
            >>> def func(x):
            >>>     return x
        """
        if func_or_class is None:
            # decorator
            def _register_func(func_or_class):
                self._registry[key] = func_or_class
                return func_or_class

            return _register_func
        else:
            self._registry[key] = func_or_class
