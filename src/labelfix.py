import multiprocessing
import time
import copy
import gzip
import os
import shutil
import warnings

from urllib.request import urlopen

import gensim
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from keras_preprocessing.image import ImageDataGenerator
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import compute_class_weight
from tensorflow.python.client import device_lib

from src.models.cnn_keras import get_model_cnn
from src.models.dense_net_keras import get_model_dense

warnings.filterwarnings("ignore", message="F-score is ill-defined and being set to 0.0 in labels with no predicted samples.")
warnings.filterwarnings("ignore", message="Data with input dtype int64 was converted to float64 by MinMaxScaler.")

gpu_avail = np.any(["gpu" in device_lib.list_local_devices()[i].name for i in range(len(device_lib.list_local_devices()))])


def _get_indices(pred, y):
    """
    For internal use: Given a prediction of a model and labels y, return sorted indices of likely mislabeled instances
    :param pred:        np array, Predictions made by a classifier
    :param y:           np array, Labels according to the data set
    :return:            np array, List of indices sorted by how likely they are wrong according to the classifier
    """
    assert pred.shape[0] == y.shape[0], "Pred {} =! y shape {}".format(pred.shape[0], y.shape[0])
    y_squeezed = y.squeeze()
    if y_squeezed.ndim == 2:
        dots = [np.dot(pred[i], y_squeezed[i]) for i in range(len(pred))]  # one-hot y
    elif y_squeezed.ndim == 1:
        dots = [pred[i, y_squeezed[i]] for i in range(len(pred))]  # numeric y
    else:
        raise ValueError("Wrong dimension of y!")
    indices = np.argsort(dots)
    return indices


def _precision_on_k(pred, y_noisy, y_true, alpha):
    """
    For internal use: Calculate the accuracy of a given prediction
    :param pred:            np array, predictions made by a classifier
    :param y_noisy:         np array, labels of the data set with added noise
    :param y_true:          np array, labels of the data set without added noise
    :param alpha:           float, percentage of most likely data points to be checked
    :return:                float, custom precision value according to equation in paper
    """
    assert 0 <= alpha <= 1
    len_examined = int(pred.shape[0] * alpha)
    # since y_noisy is one-hot, this returns certainty of classifier wrt. y_noisy label
    indices = _get_indices(pred=pred, y=y_noisy)[:len_examined]
    true_flips = np.not_equal(y_true[indices], y_noisy[indices])  # for categorical
    return np.sum(true_flips) / float(len_examined)


def _calc_recall(pred, y_noisy, y_true, alpha, mu):
    """
    For internal use: Calculate the goodness of a given prediction
    :param pred:            np array, predictions made by a classifier
    :param y_noisy:         np array, labels of the data set with added noise
    :param y_true:          np array, labels of the data set without added noise
    :param alpha:           float, percentage of most likely data points to be checked
    :return:                float, custom recall value according to equation in paper
    """
    assert 0 <= alpha <= 1
    len_examined = int(pred.shape[0] * alpha)
    # since y_noisy is one-hot, this returns certainty of classifer wrt. y_noisy label
    indices = _get_indices(pred=pred, y=y_noisy)[:len_examined]
    true_flips = np.not_equal(y_true[indices], y_noisy[indices])
    return np.sum(true_flips) / float(int(pred.shape[0] * mu))


def _get_runtime(start):
    """
    For internal use: Get runtime of training + evaluation
    :param start:       float, starting time as float given by time.time()
    :return:            str, time needed by operation in hours, minutes or seconds depending on time taken.
    """
    delta = time.time() - start
    if delta < 60:
        return str(round(delta, 1)) + " sec"
    elif delta < 60 * 60:
        return str(round(delta / 60, 2)) + " min"
    else:
        return str(round(delta / 3600, 2)) + " h"


def is_numerical(X):
    """
    Check if the the data set is most likely numerical data. Check is only based on dimensionality and not actual types.
    :param X:       np array, Data set to check
    :return:        bool, True if data is probably numerical, False otherwise
    """
    is_num = X.ndim == 2
    if is_num:
        print("Assuming numerical input since data has dimensionality {}.".format(X.ndim))
    return is_num


def is_textual(X):
    """
    Check if the the data set is most likely textual data. Check is based on strings in the data set.
    :param X:       np array, Data set to check
    :return:        bool, True if data is probably textual, False otherwise
    """
    unique_types = np.unique([str(type(X[i])) for i in range(X.shape[0])])
    is_txt = (X.ndim == 1 and np.all(["str" in unique_types[i] for i in range(unique_types.shape[0])]))
    if is_txt:
        print("Data is specified to be textual")
    return is_txt


def is_image(X):
    """
    Check if the the data set is most likely image data. Check is only based on dimensionality and not actual types.
    :param X:       np array, Data set to check
    :return:        bool, True if data is probably image data, False otherwise
    """
    is_img = X.ndim in [3, 4]
    if is_img:
        print("Assuming image input since data has dimensionality {}.".format(X.ndim))
    else:
        print("Assuming no image input since data has dimensionality {}.".format(X.ndim))
    return is_img


# noinspection PyUnreachableCode
def preprocess_x_y_and_shuffle(X, y):
    """
    Preprocess the input. Returns preprocessed tuple X', y'
    :param X:       np array, Input data
    :param y:       np array, Labels
    :return:        (np array, np array), Tuple of preprocessed input data X and labels y
    """
    # squeeze y
    y = np.asarray(y).squeeze()
    X = np.asarray(X)

    # Some sanity checks
    assert y.shape[0] == X.shape[0]
    assert X.shape[0] >= 30, "Dataset has less than 30 non-NAN values!"

    if is_textual(X):
        print("Applying textual preprocessing on data with shape {}.".format(X.shape))
        # Paths for word vector files
        word2vec_file = '../res/wordvecs/GoogleNews-vectors-negative300-SLIM.bin'
        word2vec_download_url = 'https://github.com/eyaler/word2vec-slim/raw/master/GoogleNews-vectors-negative300-SLIM.bin.gz'
        # Check if file already exists
        if not os.path.exists(word2vec_file):
            # Ensure directory exists
            if not os.path.exists(os.path.dirname(word2vec_file)):
                os.makedirs(os.path.dirname(word2vec_file))

            # Download the file and store it intermediately, so we don't have to do everything in memory.
            request = urlopen(word2vec_download_url)
            intermediate_file_path = word2vec_file+'.gz'
            with open(intermediate_file_path, 'wb') as file:
                file.write(request.read())
            # Decompress the file
            with gzip.open(intermediate_file_path, 'rb') as gzip_file:
                with open(word2vec_file, 'wb') as final_file:
                    shutil.copyfileobj(gzip_file, final_file)
            # Remove the intermediate file again
            os.remove(intermediate_file_path)

        model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_file, binary=True)
        res = []
        for j in range(X.shape[0]):
            # preprocess current line
            current_line = X[j] \
                .replace(".", "") \
                .replace(",", "") \
                .replace("!", "") \
                .replace("'", "") \
                .replace('"', "") \
                .replace("?", "").lower().split(" ")

            # if words in current line are in vocab
            if len([i for i in current_line if i in model.vocab]) > 0:
                tmp = np.sum([model[i] for i in current_line if i in model.vocab], axis=0)
            # if no words in vocab, return 0 vector
            else:
                tmp = np.asarray([0] * 300)

            res.append(list(tmp))
        X = np.asarray(res)
        del model
        return X, y

    # fallthrough from textual pipeline
    if is_numerical(X) or is_textual(X):
        min_max = MinMaxScaler(feature_range=(-1, 1))
        X = min_max.fit_transform(X)
        return X, y

    elif is_image(X):
        # add channel information if missing!
        if X.ndim == 3:
            print("Assuming image input without channel information since data has dimensionality {}. "
                  "Adding single channel to data now...".format(X.ndim))
            data_shape = X.shape
            X = X.reshape((*data_shape, 1))

        return X, y

    else:
        raise ValueError("X has dimensionality of {}! Must be either 2 (numerical)"
                         "or 3 or 4 (image with or without channel information)".format(X.ndim))


def print_statistics(X, y):
    """
    Given X and y, print various statistics to stdout
    :param X:       np array, Input of the data set
    :param y:       np array, Labels of the data set
    :return:        float, factor  (most common label)/(least common label)
    """
    print("\nDataset statistics:")
    print("Shape of X: {}, shape of y: {}".format(X.shape, y.shape))
    if y.ndim == 2:
        vc = pd.Series(np.argmax(y, axis=1)).value_counts(normalize=True)
    else:
        vc = pd.Series(y).value_counts(normalize=True)
    print("Distribution of labels: y.max_count / y.min_count: {} \n".format(vc.max() / vc.min()))
    return vc.max() / vc.min()


def check_dataset(X, y, hyperparams=None):
    """
    Check a given dataset for mislabeled instances
    :param X:               data
    :param y:               labels
    :param epochs:          epochs for which to train the classifier predicting potentially mislabeled instances
    :param hyperparams:     a dictionary with key/values to use for the CLF. If provided, there is no gridsearch!
    :return:                a dictionary containing the results, where key 'indices' returns mislabels sorted
    """
    imbalance = print_statistics(X, y)
    num_classes = np.unique(y).shape[0]
    class_weight = compute_class_weight('balanced', np.unique(y), y)
    # begin time measurement
    start = time.time()

    # ========================================== if numerical ==========================================
    if is_numerical(X) or is_textual(X):
        # Assert incoming data format
        y = y.squeeze()
        assert y.ndim == 1, "ONE HOT FOR KERAS! Text or Numeric requires numerical labels!"

        # Set up early stopping
        do_val_split = X.shape[0] > 1000
        metric = "acc"
        es = EarlyStopping(monitor="val_" + metric if do_val_split else metric,
                           min_delta=0.01, patience=5, verbose=0, mode='max', baseline=None,
                           restore_best_weights=True)

        # Grid Seach Loop
        val_split_size = 0.1
        if not hyperparams:

            fit_params = dict(callbacks=[es], class_weight=class_weight,
                              validation_split=val_split_size if do_val_split else None)

            hyperparameter_dict = {
                "input_dim": [X.shape[1]],
                "output_dim": [num_classes],
                "num_hidden": [1, 2, 3, 5],
                "size_hidden": [50, 120],
                "dropout": [0, 0.1, 0.2],
                "epochs": [400],
                "learn_rate": [1e-2],
                "activation": ["relu"]
            }

            estimator = KerasClassifier(build_fn=get_model_dense)

            num_cpus = int(multiprocessing.cpu_count() * 0.6)
            grid_search_transfer = GridSearchCV(
                estimator=estimator,
                param_grid=hyperparameter_dict,
                scoring=make_scorer(lambda y, y_pred: f1_score(y, y_pred, average='macro')),
                n_jobs=num_cpus + 1,
                verbose=0,
                cv=3,
                iid=False,
                error_score='raise',
                return_train_score=False,
                refit=False)

            grid_search_transfer.fit(X, y, verbose=0, **fit_params)
            best_params = grid_search_transfer.best_params_
            print("Best score found: {} with hyper params {}".format(grid_search_transfer.best_score_, best_params))
        else:
            best_params = hyperparams

        # Refit two estimators
        y = to_categorical(y)

        # Prepare data and hyper params
        bp = copy.deepcopy(best_params)
        del bp["epochs"]

        nn = get_model_dense(**bp)
        nn.fit(X, y, epochs=best_params["epochs"], verbose=0, callbacks=[es], class_weight=class_weight,
                validation_split=val_split_size if do_val_split else None)
        pred = nn.predict_proba(X)  # predict test set

    # ========================================== if image ==========================================
    elif is_image(X):
        X = np.asarray(X, dtype=np.float32)
        y = to_categorical(y.squeeze())

        do_val_split = True
        metric = "acc"
        es = EarlyStopping(monitor="val_" + metric if do_val_split else metric, min_delta=0.005, patience=15,
                           verbose=0, mode='max', baseline=None,
                           restore_best_weights=True)

        # Grid Search Loop
        val_split_size = 0.05
        best_params = {"Fixed sized CNN"}

        nn = get_model_cnn(shape_x=X.shape[1:], shape_y=y.shape[1:][0])
        nn.summary()
        datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, zca_whitening=False)
        datagen.fit(X)
        X, y = datagen.flow(X, y, batch_size=X.shape[0], shuffle=False).next()
        nn.fit(X, y, epochs=100, verbose=0, callbacks=[es], class_weight=class_weight, validation_split=val_split_size)
        pred = nn.predict_proba(X)  # predict test set

    else:
        raise ValueError("No branch for training and extracting was taken!")

    # calc acc
    return {"Size": X.shape,
            "Classes": np.unique(y, axis=0).shape[0],
            "runtime": _get_runtime(start),
            "pred": pred,
            "indices": _get_indices(pred=pred, y=y),
            "best_params": best_params,
            "imbalance": imbalance
            }
