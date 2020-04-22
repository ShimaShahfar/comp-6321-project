import os
import functools
import tensorflow as tf
import pydoc
import pandas as pd
from math import ceil
from base.base_engine import BaseEngine
from utils.lr_finder import LRFinder
from matplotlib import pyplot as plt


class LR_Experiment(BaseEngine):
    def __init__(self, model, data, config):
        super().__init__(model, data, config)
        self.callbacks = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        self.init_callbacks()

    def init_callbacks(self):
        if "tensorboard" in self.config.callbacks:
            self.callbacks.append(
                tf.keras.callbacks.TensorBoard(
                    log_dir=os.path.join(self.config.experiment_dir, "logs"),
                    **self.config.callbacks.tensorboard
                )
            )
        if "csv_logger" in self.config.callbacks:
            self.callbacks.append(
                tf.keras.callbacks.CSVLogger(
                    filename=os.path.join(self.config.experiment_dir, "history.csv")
                )
            )

    def run(self):
        # check GPU
        print("Device: ", tf.test.gpu_device_name())

        # load from checkpoint
        if self.config.checkpoint:
            self.model.load_weights(self.config.checkpoint, by_name=True)

        # create optimizer
        optimizer = pydoc.locate(self.config.optimizer.name)(
            **self.config.optimizer.args
        )


        # load data
        train_flow = self.data.train

        # compilation arguments
        compile_args = dict(
            loss=self.config.loss,
            metrics=[
                "accuracy"],
        )

        self.model.compile(optimizer=optimizer, **compile_args)

        epochs = ceil(self.config.iterations / len(train_flow))
        steps_per_epoch = self.config.iterations // epochs

        lr_finder = LRFinder(self.model)
        lr_finder.find_generator(
            train_flow,
            start_lr=self.config.start_lr,
            end_lr=self.config.end_lr,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=self.callbacks,
            workers=self.config.workers,
            verbose=self.config.verbose,
        )

        lr_finder.plot_loss(n_skip_beginning=10, n_skip_end=5)
        plt.savefig(
            os.path.join(
                self.config.experiment_dir,
                "loss_{}it.png".format(self.config.iterations),
            )
        )

        lr_finder.plot_loss_change(
            sma=20, n_skip_beginning=10, n_skip_end=5, y_lim=(-0.01, 0.01)
        )
        plt.savefig(
            os.path.join(
                self.config.experiment_dir,
                "loss_change_{}it.png".format(self.config.iterations),
            )
        )

        lr_finder.plot_exp_loss()
        plt.savefig(
            os.path.join(
                self.config.experiment_dir,
                "exp_loss_{}it.png".format(self.config.iterations),
            )
        )

        lr_finder.plot_exp_loss_change()
        plt.savefig(
            os.path.join(
                self.config.experiment_dir,
                "exp_loss_change_{}it.png".format(self.config.iterations),
            )
        )

        pd.DataFrame({"lr": lr_finder.lrs, "loss": lr_finder.losses}).to_csv(
            os.path.join(self.config.experiment_dir, "lr_loss.csv")
        )
