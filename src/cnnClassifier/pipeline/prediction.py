import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
        self.model_path = os.path.join("artifacts", "training", "model.keras")
        self.class_names = ["Non-Lesion", "Lesion"]

    def preprocess(self):
        image = tf.keras.utils.load_img(self.filename, target_size=(224, 224, 3))
        image = tf.keras.utils.img_to_array(image) / 255.0
        return np.expand_dims(image, axis=0)

    def predict(self):
        model = keras.models.load_model(self.model_path)
        image_array = self.preprocess()
        preds = model.predict(image_array)
        class_idx = np.argmax(preds, axis=1)[0]
        return self.class_names[class_idx]

    def predict_proba(self):
        model = keras.models.load_model(self.model_path)
        image_array = self.preprocess()
        preds = model.predict(image_array)[0]
        return {self.class_names[i]: float(preds[i]) for i in range(len(self.class_names))}
 
        