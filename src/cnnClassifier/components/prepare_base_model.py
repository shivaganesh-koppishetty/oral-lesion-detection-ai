import os
import tensorflow as tf
from pathlib import Path
from cnnClassifier import logger
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
        logger.info("PrepareBaseModel: Initialized with configuration")

    def get_base_model(self):
        try:
            logger.info("get_base_model: started")

            self.model = tf.keras.applications.VGG16(
                input_shape=self.config.params_image_size,
                weights=self.config.params_weights,
                include_top=self.config.params_include_top
            )

            logger.info("Base model loaded successfully")
            self.save_model(path=self.config.base_model_path, model=self.model)
            logger.info("get_base_model: completed")

        except Exception:
            logger.exception("Exception occurred in get_base_model")
            raise

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        try:
            logger.info("_prepare_full_model: started")

            if freeze_all:
                for layer in model.layers:
                    layer.trainable = False
                

            elif (freeze_till is not None) and (freeze_till > 0):
                for layer in model.layers[: -freeze_till]:
                    layer.trainable = False
                

            flatten_in = tf.keras.layers.Flatten()(model.output)
            prediction = tf.keras.layers.Dense(units=classes, activation="softmax")(flatten_in)

            full_model = tf.keras.Model(
                inputs=model.input,
                outputs=prediction
            )

            full_model.compile(
                optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=["Accuracy"]
            )

            full_model.summary()
            logger.info("_prepare_full_model: completed")

            return full_model

        except Exception:
            logger.exception("Exception occurred in _prepare_full_model")
            raise

    def update_base_model(self):
        try:
            logger.info("update_base_model: started")

            self.full_model = self._prepare_full_model(
                model=self.model,
                classes=self.config.params_classes,
                freeze_all=True,
                freeze_till=None,
                learning_rate=self.config.params_learning_rate
            )

            self.save_model(path=self.config.updated_base_model_path, model=self.full_model)
            logger.info("update_base_model: completed")

        except Exception:
            logger.exception("Exception occurred in update_base_model")
            raise

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        try:
            model.save(path)
            logger.info(f"Model saved at: {path}")
        except Exception:
            logger.exception(f"Exception occurred while saving model to {path}")
            raise
        