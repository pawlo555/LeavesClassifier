import tensorflow as tf


class Dataset:
    def __init__(self):
        self.train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            directory="Dataset/training", label_mode='categorical', seed=123)
        self.validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            directory="Dataset/validating", label_mode='categorical', seed=123)
        self.test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            directory="Dataset/testing", label_mode='categorical', seed=123)
