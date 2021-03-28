import tensorflow as tf

recover = lambda x,y: y


def is_train(x, y):
    return x % 10 < 8


def is_validation(x, y):
    return x % 10 == 8


def is_test(x, y):
    return x % 10 == 9


def split(full_dataset):
    train_dataset = full_dataset.enumerate().filter(is_train).map(recover)
    validation_dataset = full_dataset.enumerate().filter(is_validation).map(recover)
    test_dataset = full_dataset.enumerate().filter(is_test).map(recover)
    return train_dataset, validation_dataset, test_dataset


class Dataset:
    def __init__(self, greyscale=True, segmented=True):
        full_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            directory='PlantVillage-Dataset/raw/color', label_mode='categorical', seed=123)
        self.train_dataset = full_dataset.enumerate().filter(is_train).map(recover)
        self.validation_dataset = full_dataset.enumerate().filter(is_validation).map(recover)
        self.test_dataset = full_dataset.enumerate().filter(is_test).map(recover)
        if greyscale:
            self.add_photos('PlantVillage-Dataset/raw/grayscale')
        if segmented:
            self.add_photos('PlantVillage-Dataset/raw/segmented')


    def add_photos(self, path):
        full = tf.keras.preprocessing.image_dataset_from_directory(
            directory=path, label_mode='categorical', seed=123)
        train, validation, test = split(full)
        self.train_dataset = self.train_dataset.concatenate(train)
        self.validation_dataset = self.train_dataset.concatenate(validation)
        self.test_dataset = self.train_dataset.concatenate(test)

my_dataset = Dataset()
print(my_dataset.test_dataset)

inputs = tf.keras.Input(shape=(256, 256, 3,))
x = tf.keras.layers.Flatten()(inputs)
x = tf.keras.layers.Dense(38, activation=tf.nn.relu)(x)
outputs = tf.keras.layers.Dense(38, activation=tf.nn.softmax)(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.summary()
model.compile(optimizer="Adam", loss="mse")
model.fit(my_dataset.train_dataset)
