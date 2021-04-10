import tensorflow as tf

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
# model.evaluate(my_dataset.train_dataset)
