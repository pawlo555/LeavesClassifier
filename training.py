import tensorboard
import tensorflow as tf
from dataset_loader import Dataset
from models import get_cnn_model, get_inception_model
import datetime


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


my_dataset = Dataset(greyscale=False, segmented=False)
print(my_dataset.test_dataset)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


model = get_inception_model()
model.summary()
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=1800,
    decay_rate=0.95)

optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, nesterov=True)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(my_dataset.train_dataset, epochs=30)
model.save("models/inception_01")
