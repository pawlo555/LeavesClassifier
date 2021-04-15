import tensorflow as tf
import os

from dataset_loader import Dataset
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

val_dataset = Dataset().validation_dataset

for directory in os.listdir("models"):
    print(directory)
    model = tf.keras.models.load_model("models/" + directory)
    model.evaluate(val_dataset)
