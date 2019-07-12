# Function to create model, required for KerasClassifier
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD


def get_model_dense(input_dim, output_dim, num_hidden, size_hidden, learn_rate, activation='relu', dropout=0):
    """
    Create a new dense model for training and classification.
    :param input_dim:       int, Dimension of input layer
    :param output_dim:      int, Dimension of output layer
    :param num_hidden:      int, Number of hidden layers
    :param size_hidden:     int, Number of neurons in each hidden layer
    :param learn_rate:      float, Learning rate for training
    :param activation:      str or keras activation function, Activation function used in the network
    :param dropout:         float, Probability of drop for dropout
    :return:                keras model, A compiled Keras model ready for training
    """
    # create model

    model = Sequential()
    model.add(Dense(size_hidden, input_dim=input_dim, activation=activation))
    model.add(Dropout(dropout))
    for i in range(num_hidden - 1):
        model.add(Dense(size_hidden, activation=activation))
        model.add(Dropout(dropout))
    model.add(Dense(output_dim, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=learn_rate), metrics=['accuracy'])
    return model
