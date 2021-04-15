import tensorboard
import tensorflow as tf
from dataset_loader import Dataset
from models import get_cnn_model, get_inception_model, get_residual_model


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


model = tf.keras.models.load_model("models/residual_01")
model.summary()
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=5000,
    decay_rate=0.95)

optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, nesterov=True)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(Dataset().train_dataset, epochs=10)
model.save("models/residual_02")
