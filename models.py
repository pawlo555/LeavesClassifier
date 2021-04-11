import tensorflow as tf


def get_cnn_model():
    inputs = tf.keras.Input(shape=(256, 256, 3,))
    x = tf.keras.layers.Conv2D(8, (3, 3), padding='same')(inputs)
    x = tf.keras.layers.Conv2D(16, (3, 3), padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=1500, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(units=1000, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(units=500, activation='relu')(x)
    outputs = tf.keras.layers.Dense(units=38, activation='softmax')(x)
    cnn_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return cnn_model


def get_inception_model():
    inputs = tf.keras.Input(shape=(256, 256, 3,))
    x = tf.keras.layers.Conv2D(32, (7, 7), strides=(2, 2), padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(1, 1))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((3, 3))(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = inception_layer(x, (64, 128, 32, 32, 96, 16))
    x = inception_layer(x, (128, 196, 96, 64, 128, 32))
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = inception_layer(x, (192, 208, 48, 64, 96, 16))
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = inception_layer(x, (256, 320, 128, 128, 160, 32))
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = inception_layer(x, (256, 320, 128, 128, 160, 32))

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=1000, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(units=38, activation='softmax')(x)
    inception_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return inception_model


def get_residual_model():
    inputs = tf.keras.Input(shape=(256, 256, 3,))
    x = tf.keras.layers.Conv2D(8, (3, 3), padding='same')(inputs)
    x = tf.keras.layers.Conv2D(16, (3, 3), padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    for i in range(4):
        for _ in range(4):
            x = residual_layer(x, 2 ** i * 32)
        x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=1000, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(units=38, activation='softmax')(x)
    inception_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return inception_model


def inception_layer(input, filters):
    """
    Creates a inception layer
    :param input:
    :param filters: tuple with numbers of filters in inception layer
    :return:
    """
    conv_1 = tf.keras.layers.Conv2D(filters[0], (1, 1), padding='same')(input)
    conv_3 = tf.keras.layers.Conv2D(filters[4], (1, 1), padding='same')(input)
    conv_3 = tf.keras.layers.Conv2D(filters[1], (3, 3), padding='same')(conv_3)
    conv_5 = tf.keras.layers.Conv2D(filters[5], (1, 1), padding='same')(input)
    conv_5 = tf.keras.layers.Conv2D(filters[2], (5, 5), padding='same')(conv_5)
    max_pool = tf.keras.layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input)
    max_pool = tf.keras.layers.Conv2D(filters[3], (1, 1), padding='same')(max_pool)
    output = tf.keras.layers.concatenate([conv_1, conv_3, conv_5, max_pool], axis=3)
    return output


def residual_layer(input, filters):
    """
    Residual layer
    :param input:
    :param filters:
    :return: output of residual layer
    """
    x = tf.keras.layers.Conv2D(filters, kernel_size=(3, 3), padding='same')(input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filters, kernel_size=(3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    last_dim = tf.keras.backend.int_shape(input)
    if last_dim[3] != filters:
        input = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), padding='same')(input)
    added = tf.keras.layers.add([input, x])
    return tf.keras.layers.Activation('relu')(added)
