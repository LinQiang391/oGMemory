class NoModel:
    def __getattr__(self, name):
        raise AssertionError(f"Recovery must not call a model: {name}")
