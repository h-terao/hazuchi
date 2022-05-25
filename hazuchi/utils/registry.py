import functools


class Registry(dict):
    """A registry class to access function or class by the hashable keys.

    Example:
        >>> registry = Registry()
        >>> @registry.register("add")
        >>> @registry.register("add_one", y=1)
        >>> def add(x, y):
        >>>     return x + y
        >>> registry["add"](2, 3)  # compute 2+3
        2
        >>> registry["add_one"](4)  # 4+1
        4
    """

    def register(self, key, **kwargs):
        """Register a function or class.

        Args:
            key: A hashable object to register the registry.
            kwargs: Default arguments.
        """

        def wrap(func_or_class):
            self[key] = functools.partial(func_or_class, **kwargs)
            return func_or_class

        return wrap
