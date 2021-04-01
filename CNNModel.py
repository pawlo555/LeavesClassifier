import tensorflow as tf
from dataset_loader import Dataset

my_dataset = Dataset(greyscale=False, segmented=False)
print(my_dataset.test_dataset)

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass


def get_cnn_model():
    inputs = tf.keras.Input(shape=(256, 256, 3,))
    x = tf.keras.layers.Conv2D(8, (3, 3), padding='same')(inputs)
    x = tf.keras.layers.Conv2D(16, (3, 3), padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
    #x = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
    #x = tf.keras.layers.Conv2D(256, (3, 3), padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
    #x = tf.keras.layers.Conv2D(512, (3, 3), padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    #x = tf.keras.layers.Dense(units=1500, activation='relu')(x)
    #x = tf.keras.layers.Dense(units=1000, activation='relu')(x)
    x = tf.keras.layers.Dense(units=500, activation='relu')(x)
    outputs = tf.keras.layers.Dense(units=38, activation='softmax')(x)
    cnn_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return cnn_model


model = get_cnn_model()
model.summary()
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(my_dataset.train_dataset, epochs=10, batch_size=16)
