import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

from dataset_loader import Dataset
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


my_dataset = Dataset(greyscale=False, segmented=False)
model = tf.keras.models.load_model("models/cnn_model_02")
#model.evaluate(my_dataset.train_dataset)

# My attempt to predict

# print("Predicting...")
# sample_image = image.load_img("AppleScab.JPG")
# input_arr = image.img_to_array(sample_image)
# input_arr = np.array([input_arr])   # Convert single image to a batch
# predictions = model.predict(input_arr)
# print(predictions)
# print("Predicted!")



