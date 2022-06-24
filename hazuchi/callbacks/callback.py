from flax.serialization import register_serialization_state


PRIORITY_WRITER = 300
PRIORITY_EDITOR = 200
PRIORITY_READER = 100
PRIORITY_SNAPSHOT = -100


class Callback:
    """Base of callback object."""

    priority: int = PRIORITY_READER

    def on_fit_start(self, trainer, train_state):
        return train_state

    def on_fit_epoch_start(self, trainer, train_state):
        return train_state

    def on_fit_epoch_end(self, trainer, train_state, summary):
        return train_state, summary

    def on_fit_end(self, trainer, train_state):
        return train_state

    def on_test_start(self, trainer, train_state):
        return train_state

    def on_test_epoch_start(self, trainer, train_state):
        return train_state

    def on_test_epoch_end(self, trainer, train_state, summary):
        return train_state, summary

    def on_test_end(self, trainer, train_state):
        return train_state

    def to_state_dict(self):
        return {}

    def from_state_dict(self, state):
        return self

    def finalize(self):
        pass


register_serialization_state(
    Callback,
    ty_to_state_dict=lambda x: x.to_state_dict(),
    ty_from_state_dict=lambda x, s: x.from_state_dict(s),
)
