import functools


class Registry(dict):
    """A simple registry class.

    Examples:
        >>> registry = Registry()
        >>> @registry.register("add")
        >>> def add(a, b):
        >>>     return a + b
        >>> registry["multiply"] = lambda a, b: a*b
        >>> sorted(registry)
        ["add", "multiply"]
        >>> registry.add(1, 2)
        3
        >>> registry["add"](1, 2)
        3
        >>> registry.multiply(1, 2)
        2
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

    def register(self, key: str, /, **kwargs):
        if key in self:
            raise ValueError(
                f"{key} is already registered. Use different keys to register func or class."
            )

        def deco(func_or_class):
            if kwargs:
                self[key] = functools.partial(func_or_class, **kwargs)
            else:
                self[key] = func_or_class
            return func_or_class

        return deco
