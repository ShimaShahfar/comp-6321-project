import pydoc
from base.base_data_loader import BaseDataLoader


class DataLoader(BaseDataLoader):
    def __init__(self, config):
        super().__init__(config)

        self.datagen = pydoc.locate("data_loader." + self.config.datagen_name)
        self.datagen_configs = {
            name: dict(self.config[name + "_args"], **self.config["base_args"])
            for name in ["train", "val", "test"]
        }

    @property
    def train(self):
        return self.datagen(**self.datagen_configs["train"])

    @property
    def val(self):
        return self.datagen(**self.datagen_configs["val"])

    @property
    def test(self):
        return self.datagen(**self.datagen_configs["test"])
