class BaseDataLoader(object):
    def __init__(self, config):
        self.config = config

    @property
    def train(self):
        raise NotImplementedError

    @property
    def val(self):
        raise NotImplementedError

    @property
    def test(self):
        raise NotImplementedError