import tensorflow as tf
from dataset_loader import Dataset
from CNNModel import get_cnn_model

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

my_dataset = Dataset(greyscale=False, segmented=False)
print(my_dataset.test_dataset)

model = get_cnn_model()
model.summary()
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=10000,
    decay_rate=0.9)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, nesterov=True)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(my_dataset.train_dataset, epochs=1)
model.save("first_cnn_model_1ep")
model.evaluate(my_dataset.train_dataset)
