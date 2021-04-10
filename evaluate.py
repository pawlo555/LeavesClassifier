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

for root, dirs, files in os.walk(r"C:\Studia\Biologiczne\LeavesClassifier\PlantVillage-Dataset\raw\color", topdown=False):
   for name in dirs:
      print(os.path.join(name))


my_dataset = Dataset(greyscale=False, segmented=False)
model = tf.keras.models.load_model("models/inception_01")
# model.evaluate(my_dataset.train_dataset)
