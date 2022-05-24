class Registry:
    def __init__(self):
        self._memory = {}

    def register(self, key, overwrite: bool = False):
        """
        Example:
            >>> model_registry = Registry()
            >>> @model_registry.register(name=resnet18)
            >>> class ResNet()
        """

        def _register(func_or_class):
            if key in self._memory and not overwrite:
                raise ValueError(f"{key} is already registered.")
            self._memory[key] = func_or_class

        return _register

    def keys(self):
        return self._memory.keys()

    def values(self):
        return self._memory.values()

    def items(self):
        return self._memory.items()

    def __getitem__(self, key):
        return self._memory[key]

    def __contains__(self, key):
        return key in self._memory

    def __iter__(self):
        return self._memory.keys()
