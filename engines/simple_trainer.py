import os
import pydoc
import tensorflow as tf
from base.base_engine import BaseEngine


class SimpleUNetTrainer(BaseEngine):
    def __init__(self, model, data, config):
        super().__init__(model, data, config)
        self.callbacks = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        self.init_callbacks()


    def init_callbacks(self):
        if "checkpoint" in self.config.callbacks:
            chkp_dir = os.path.join(self.config.experiment_dir, "checkpoints")
            os.makedirs(chkp_dir, exist_ok=True)  # create dirs
            self.callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join(
                        chkp_dir,
                        "dataweights-epoch-{{epoch:02d}}-{0}-{{{0}:.2f}}.hdf5".format(
                            self.config.callbacks.checkpoint.monitor
                        ),
                    ),
                    **self.config.callbacks.checkpoint
                )
            )
        if "tensorboard" in self.config.callbacks:
            self.callbacks.append(
                tf.keras.callbacks.TensorBoard(
                    log_dir=os.path.join(self.config.experiment_dir, "logs"),
                    **self.config.callbacks.tensorboard
                )
            )

    def run(self):
        if self.config.checkpoint:
            print("Loading from checkpoint....")
            self.model.load_weights(self.config.checkpoint, by_name=True)

        optimizer = pydoc.locate(self.config.optimizer.name)(
            **self.config.optimizer.args.toDict()
        )

        self.model.compile(
            optimizer=optimizer,
            loss=self.config.loss,
            metrics=["accuracy"]
        )

        train_flow = self.data.train
        val_flow = self.data.val
 
        print("Device: ", tf.test.gpu_device_name())
        self.model.fit_generator(
            train_flow,
            validation_data=val_flow,
            epochs=self.config.num_epochs,
            # steps_per_epoch=self.config.steps_per_epoch,
            verbose=self.config.verbose,
            workers=self.config.workers,
            callbacks=self.callbacks,
        )

        self.model.save(os.path.join(self.config.experiment_dir, "final.h5"))