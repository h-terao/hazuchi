PRIORITY_WRITER = 300
PRIORITY_EDITOR = 200
PRIORITY_READER = 100
PRIORITY_SNAPSHOT = -100


class Callback:
    """
    NOTE:
        ```python
        on_train_start()
        for epoch in range(max_epochs):
            on_epoch_start()

            on_train_epoch_start()
            for batch in train_data:
                on_train_step_start()
                update_model
                on_train_step_end()
            on_train_epoch_end()

            on_val_epoch_start()
            for batch in train_data:
                on_val_step_start()
                validate model
                on_val_step_end()
            on_val_epoch_end()

            on_epoch_end()
        on_train_end()
        ```
    """

    priority: int = PRIORITY_READER

    def on_fit_start(self, trainer, train_state):
        return train_state

    def on_fit_epoch_start(self, trainer, train_state):
        return train_state

    def on_train_epoch_start(self, trainer, train_state):
        return train_state

    def on_train_step_start(self, trainer, train_state):
        return train_state

    def on_train_step_end(self, trainer, train_state, summary):
        return train_state, summary

    def on_train_epoch_end(self, trainer, train_state, summary):
        return train_state, summary

    def on_val_epoch_start(self, trainer, train_state):
        return train_state

    def on_val_step_start(self, trainer, train_state):
        return train_state

    def on_val_step_end(self, trainer, train_state, summary):
        return train_state, summary

    def on_val_epoch_end(self, trainer, train_state, summary):
        return train_state, summary

    def on_fit_epoch_end(self, trainer, train_state, summary):
        return train_state, summary

    def on_fit_end(self, trainer, train_state):
        return train_state

    def on_test_start(self, trainer, train_state):
        return train_state

    def on_test_epoch_start(self, trainer, train_state):
        return train_state

    def on_test_step_start(self, trainer, train_state):
        return train_state

    def on_test_step_end(self, trainer, train_state, summary):
        return train_state, summary

    def on_test_epoch_end(self, trainer, train_state, summary):
        return train_state, summary

    def on_test_end(self, trainer, train_state, summary):
        return train_state, summary

    def to_state_dict(self):
        return {}

    def from_state_dict(self, state) -> None:
        # do nothing in default.
        return None
