# Function to create model, required for KerasClassifier
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.optimizers import SGD


def get_model_cnn(shape_x, shape_y):
    """
    Get a simple CNN for image estimation.
    :param shape_x:         tuple of ints, Shape of a single image
    :param shape_y:         tuple of ints, Shape of the label
    :return:                Keras model, The compiled model ready for training
    """
    model = Sequential()
    model.add(Conv2D(48, (2, 2), padding='same', input_shape=shape_x, activation="relu"))
    model.add(Conv2D(48, (2, 2), activation="relu"))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))

    model.add(Conv2D(96, (2, 2), padding='same', activation="relu"))
    model.add(Conv2D(96, (2, 2), activation="relu"))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(shape_y, activation='softmax'))

    # ============================================================
    sgd = SGD(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model
