import pickle

import copy

from urllib.request import urlopen
import tarfile
from zipfile import ZipFile

import numpy as np
import os
import pandas as pd
import random

from scipy.io import arff
from keras.datasets import cifar100, cifar10, imdb
from keras_preprocessing.text import Tokenizer
from sklearn.datasets import fetch_20newsgroups, make_blobs, fetch_covtype
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

from src.labelfix import preprocess_x_y_and_shuffle


def maybe_download_dataset(file_name, url, url_file_name, extract_file=None):
    """
    Checks if a dataset of name file_name exists and tries to download it from url otherwise.
    :param file_name:       str, Name of the file of the dataset.
    :param url:             str, URL from which the dataset will be downloaded if it does not exist.
    :param url_file_name:   str, Name of the file in the URL.
    :param extract_file:    str, In case we download a zip file, we extract this file from the zip
    :return:                None
    """
    if not os.path.isfile(file_name):
        dataset_directory = os.path.dirname(file_name)

        # Ensure the directory exists
        if not os.path.exists(dataset_directory):
            os.makedirs(dataset_directory)

        # Download content from url
        response = urlopen(url + url_file_name)

        # Extract data, if we have a tar file
        if url_file_name.endswith("tar.gz") or url_file_name.endswith(".tgz"):
            intermediate_file_path = os.path.join(dataset_directory, url_file_name)
            with open(intermediate_file_path, 'wb') as intermediate_file:
                intermediate_file.write(response.read())
            tar = tarfile.open(intermediate_file_path, "r:gz")
            tar.extractall(dataset_directory)
            tar.close()
            os.remove(intermediate_file_path)
        # Extract da if we have a zip file
        elif url_file_name.endswith('.zip'):
            intermediate_file_path = os.path.join(dataset_directory, url_file_name)
            with open(intermediate_file_path, 'wb') as intermediate_file:
                intermediate_file.write(response.read())
            zip = ZipFile(intermediate_file_path, 'r')
            zip.extract(extract_file, path=dataset_directory)
            os.rename(os.path.join(dataset_directory, extract_file), file_name)
            os.remove(intermediate_file_path)
        # Otherwise we assume we already downloaded the required file
        else:
            with open(file_name, 'wb') as file:
                file.write(response.read())


def preprocess_and_noise(dataset, mu, switch_dict=None, create_test_set=False):
    """
    Given a dataset as dictionary {"data": X, "target": y}, apply noise and return with added key "target_noisy".
    :param dataset:             a dictionary {"data": X, "target": y}
    :param mu:                  fraction of noise added, float between 0 and 1
    :param switch_dict:         A dictionary, whose values are arrays of classes. Only jumble within these
    :param create_test_set:     If true, create a test set with 0 noise and save to labels "data_test", "target_test".
    :return:                    dataset augmented by key 'target_noisy' and possibly the test set
    """

    # sanity check
    if mu < 0:
        print("WARNING: MU IS LESS THAN 0 ... ADJUSTING")
        mu = 0
    elif 1 < mu:
        raise ValueError("MU={}".format(mu))
    assert 0 <= mu <= 1, "mu is {}".format(mu)

    print("Applying noise (mu={})".format(mu))

    # preprocess, import to do here before train/test split!
    dataset["data"], dataset["target"] = preprocess_x_y_and_shuffle(np.asarray(dataset["data"]), dataset["target"])

    assert dataset["target"].shape[0] == dataset["data"].shape[0]

    if create_test_set:
        # make 90 of the data the regular 'data' on which to add noise and 10 percent the post-screening evaluation data
        X_train, X_test, y_train, y_test = train_test_split(dataset["data"], dataset["target"], random_state=42,
                                                            stratify=dataset["target"], test_size=0.1)
        # assert stratify
        assert len(np.unique(y_train)) == len(np.unique(y_test)), \
            "In train: {}, in test: {}, Counts: {}".format(np.unique(y_train), np.unique(y_test),
                                                           np.bincount(dataset["target"]))

        dataset["data"] = X_train
        dataset["target"] = y_train
        dataset["data_test"] = X_test
        dataset["target_test"] = y_test

    # print jumble type
    if switch_dict is not None:
        print("Jumbling 'at random'")

    def _jumble_label(unique_labels, changed_label):
        """
        Internal helper to add noise to labels
        """
        # jumble completely at random
        if switch_dict is None:
            possible_labels = copy.deepcopy(unique_labels)
            possible_labels_removed = np.asarray([x for x in possible_labels if x != changed_label])
            assert possible_labels.shape[0] == possible_labels_removed.shape[0] + 1
            return random.choice(possible_labels_removed)
        # jumble at random
        else:
            # use argmax to check against switch_dict
            pl_list = [x for x in switch_dict.values() if changed_label in x]
            assert len(pl_list) == 1, "Error with dict! " \
                                      "Values must be encoded as ints and each value must pertain to only one group!"
            possible_labels = np.asarray(pl_list[0])
            possible_labels_removed = np.asarray([x for x in possible_labels if x != changed_label])
            return random.choice(possible_labels_removed)

    labels_unique = np.unique(dataset["target"], axis=0)
    # use stratified train_test split to add stratified noise - e.g. add the same percentage 'mu' on all classes
    old_X_shape = dataset["data"].shape
    old_y_shape = dataset["target"].shape
    X_keep, X_add_noise, y_keep, y_add_noise = train_test_split(dataset["data"], dataset["target"], random_state=42,
                                                                stratify=dataset["target"],
                                                                test_size=mu)
    y_noise_added = np.asarray([_jumble_label(labels_unique, y_add_noise[i]) for i in range(y_add_noise.shape[0])])

    # permute array, e.g. shuffle noisy examples
    permutation = np.random.permutation(old_X_shape[0])
    dataset["target_noisy"] = _permute(np.concatenate((y_keep, y_noise_added), axis=0), permutation)
    dataset["target"] = _permute(np.concatenate((y_keep, y_add_noise), axis=0), permutation)
    dataset["data"] = _permute(np.concatenate((X_keep, X_add_noise), axis=0), permutation)

    # assert that everything was done okay
    assert dataset["data"].shape[0] == X_keep.shape[0] + X_add_noise.shape[0]
    assert dataset["data"].shape[0] == old_X_shape[0]
    assert dataset["data"].shape[1] == old_X_shape[1]
    assert dataset["target"].shape[0] == old_y_shape[0]
    assert dataset["target_noisy"].shape[0] == old_y_shape[0]

    # if no noise
    if mu <= 0:
        dataset["target_noisy"] = dataset["target"]

    return dataset


# TODO doctest
def _permute(x, perm):
    assert x.shape[0] == perm.shape[0]
    res = [x[i] for i in perm]
    return np.asarray(res)


def load_toy_cov():
    """
    Loads and returns 10% of the covertype dataset. This is done using sklearn. Also the labels get adapted.
    :return:            closure, A closure function providing the dataset on call.
    """

    # Define the closure function
    def closure(mu):
        ds = fetch_covtype()
        X, _, y, _ = train_test_split(ds["data"], ds["target"], random_state=42,
                                      test_size=0.9, stratify=ds["target"])
        y = y - 1
        return preprocess_and_noise({"data": X, "target": y}, mu=mu)

    return closure

def load_toy(dataset_function):
    """
    Given a function, return a function which takes the output of dataset_function and applies noise.
    Useful for the sklearn toy datasets.
    :param dataset_function:            A function returning a data set
    :return:                            closure, A closure function providing the dataset on call.
    """

    # Define the closure function
    def closure(mu):
        ds = dataset_function()
        return preprocess_and_noise(ds, mu=mu)

    return closure


def load_adult():
    """
    Helper to load the adult data set. If dataset is not on disk, we download it from the internet.
    :return:            closure, A closure function providing the dataset on call.
    """
    # Load the dataset if necessary
    maybe_download_dataset(file_name="../res/adult/adults.csv",
                           url="http://archive.ics.uci.edu/ml/machine-learning-databases/adult/",
                           url_file_name="adult.data")

    # Define the closure function
    def closure(mu):
        df = pd.read_csv("../res/adult/adults.csv", header=None)
        for i in [1, 3, 5, 6, 7, 8, 9, 13, 14]:
            encoder = LabelEncoder()
            df[i] = encoder.fit_transform(df[i])

        target = df[14]
        df = df.drop([14], axis=1)
        ds = {"data": df.values, "target": target}
        return preprocess_and_noise(dataset=ds, mu=mu)

    return closure


def synth_classification(n_samples, n_features, n_classes, n_informative, n_redundant, n_repeated, use_blob=False):
    """
    Helper to create a synthetic data set with n_samples, n_features and n_classes, using sklearn's
    make_classification. Redundant and repeated features are added.
    :return:                            closure, A closure function providing the dataset on call.
    """

    # Define the closure function
    def closure(mu):
        if use_blob:
            a = make_blobs(n_features=n_features, n_samples=n_samples, centers=n_classes)
        else:
            a = make_classification(n_samples=n_samples,
                                    n_features=n_features,
                                    n_classes=n_classes,
                                    n_informative=n_informative,
                                    n_redundant=n_redundant,
                                    n_repeated=n_repeated,
                                    n_clusters_per_class=1,
                                    flip_y=0
                                    )
        ds = {"data": a[0], "target": a[1]}
        return preprocess_and_noise(dataset=ds, mu=mu)

    return closure


def load_cifar(num, subsection=None, switch_dict=None):
    """
    Load the cifar10 or cifar100 data set and apply noise. We use keras for loading the dataset and in case of cifar100
    res/cifar100/cifar100_names to look up the names of the labels. This is done to ease access since keras does not
    allow to read a mapping of class name to class number.
    :param num:         either 10 or 100
    :param subsection:  Either None if full data set is to be used, or array of classes to be used.
                        For example, if subselection = [0, 1, 2, 3, 4], only select 'acquiatic mammals'
    :param switch_dict: A dictionary, whose values are arrays of classes (encoded as ints). Only jumble within these
    :return:            closure, A closure function providing the dataset on call.
    """

    # Define the closure function
    def closure(mu):
        # assert noise is percentage
        assert 0 <= mu <= 1

        # load data
        if num == 100:
            (x_train, y_train), _ = cifar100.load_data(label_mode='fine')
        elif num == 10:
            (x_train, y_train), _ = cifar10.load_data()
        else:
            raise NotImplementedError("Load with 10 or 100 classes!")

        y_train = np.squeeze(y_train)

        # load names
        if num == 100:
            cifar100_names_path = "../res/cifar100/cifar100_names"
            with open(cifar100_names_path, 'rb') as fo:
                dic = pickle.load(fo, encoding='bytes')
            label_names = dic[b"fine_label_names"]
        else:
            label_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

        # if we want a subsection, e.g. [0, 1, 2, 3, 4] for acquatic mammales
        if None is not subsection:
            x_train = [x_train[i] for i in range(x_train.shape[0]) if y_train[i] in subsection]
            y_train = [y_train[i] for i in range(y_train.shape[0]) if y_train[i] in subsection]
            # to NP array. Encode values because labels need to be ints starting from 0
            encoder = LabelEncoder().fit(y_train)
            x_train = np.asarray(x_train)
            y_train = encoder.transform(np.asarray(y_train))

        ds = {"data": x_train, "target": y_train, "names": label_names}

        return preprocess_and_noise(dataset=ds, mu=mu, switch_dict=switch_dict)

    return closure


def load_fashion_mnist():
    """
    Load the fashion mnist data set.
    This is done using keras functions, so we don't have to implement our own approach.
    Also, this result in the dataset being stored in the default keras location and not the res-folder.
    :return:            closure, A closure function providing the dataset on call.
    """
    # Define the closure
    def closure(mu):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        label_names = ["top/shirt", "trousers", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag",
                       "ankle boot"]
        ds = {"data": x_train, "target": y_train, "names": label_names}
        return preprocess_and_noise(dataset=ds, mu=mu)

    return closure


def load_mnist():
    """
    Load the mnist data set.
    This is done using keras functions, so we don't have to implement our own approach.
    Also, this result in the dataset being stored in the default keras location and not the res-folder.
    :return:            closure, A closure function providing the dataset on call.
    """
    # Define the closure
    def closure(mu):
        (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()

        x_train = x_train.reshape(-1, 28, 28, 1)

        label_names = [str(x) for x in range(0, 10)]
        ds = {"data": x_train, "target": y_train, "names": label_names}
        return preprocess_and_noise(dataset=ds, mu=mu)

    return closure


def load_svhn():
    """
    Load the SVHN data set. We only need the training data.
    :return:            closure, A closure function providing the dataset on call.
    """
    # Load the dataset if necessary
    maybe_download_dataset(file_name="../res/svhn/svhn_train_32x32.mat", url="http://ufldl.stanford.edu/housenumbers/",
                           url_file_name="train_32x32.mat")

    # Define the closure function
    def closure(mu):
        from scipy import io as sio
        mat_train = sio.loadmat('../res/svhn/svhn_train_32x32.mat')
        x_train = np.moveaxis(mat_train["X"], 3, 0)  # axes need reordering
        y_train = mat_train["y"]

        # 10 classes, 1 for each digit. Digit '1' has label 1, '9' has label 9 and '0' has label 10.
        # This confuses numerical encoding - thus, subtract one
        ds = {"data": x_train, "target": y_train.squeeze() - 1, "names": [str(x) for x in range(0, 10)]}
        return preprocess_and_noise(dataset=ds, mu=mu)

    return closure


def load_ccd():
    """
    Load the credit card default data set
    :return:            closure, A closure function providing the dataset on call.
    """
    # Load the dataset if necessary
    maybe_download_dataset(file_name="../res/DefaultCreditCard/DCC.xls",
                           url="http://archive.ics.uci.edu/ml/machine-learning-databases/00350/",
                           url_file_name="default%20of%20credit%20card%20clients.xls")

    # Define the closure function
    def closure(mu):
        df = pd.read_excel("../res/DefaultCreditCard/DCC.xls", header=1, index_col=0)#, sep="\t")
        df = df.dropna(axis=0)
        target = df["default payment next month"]
        data = df.drop(["default payment next month"], axis=1).values
        label_names = ["0", "1"]

        ds = {"data": data, "target": target.values, "names": label_names}
        return preprocess_and_noise(dataset=ds, mu=mu)

    return closure


def load_twenty_newsgroup():
    """
    Load the twenty newsgroup data set using sklearn.
    :return:            closure, A closure function providing the dataset on call.
    """
    # Define the closure function
    def closure(mu):
        twenty_newsgroup = fetch_20newsgroups(subset="all", shuffle=False)

        # construct dictionary
        ds = {"data": twenty_newsgroup["data"],
              "target": twenty_newsgroup["target"],
              "names": twenty_newsgroup["target_names"]}

        # Apply noise and return
        res = preprocess_and_noise(dataset=ds, mu=mu)
        return res

    return closure


def load_imdb():
    """
    Load the IMDB dataset using keras.
    :return:            closure, A closure function providing the dataset on call.
    """
    # Define the closure function
    def closure(mu):
        (x_train, y_train), (_, _) = imdb.load_data()
        tokenizer = Tokenizer(num_words=5000)
        tokenizer.fit_on_sequences(x_train)
        x_train = tokenizer.sequences_to_matrix(x_train, "tfidf")
        # Note: svd_solver=full is needed on GPU server
        x_train = PCA(n_components=100, svd_solver='full').fit_transform(x_train)
        ds = {"data": x_train, "target": y_train}

        # Apply noise and return
        res = preprocess_and_noise(dataset=ds, mu=mu)
        return res

    return closure


def load_sms_spam():
    """
    Loads the SMS Spam Collection Dataset. The version used here is the one found on kaggle, but downloaded from a
    github repository (https://github.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/blob/master/spam.csv) since
    kaggle requires authentication for data set download.
    :return:            closure, A closure function providing the dataset on call.
    """
    # Load the dataset if necessary
    maybe_download_dataset(file_name="../res/sms_spam/sms_spam.csv",
                           url="https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/",
                           url_file_name="spam.csv")
    # Define the closure function
    def closure(mu):
        df = pd.read_csv("../res/sms_spam/sms_spam.csv", encoding="latin-1")
        df = df[["v1", "v2"]].dropna(axis=0)

        x_train = df["v2"]
        y_train = [0 if x == "ham" else 1 for x in df["v1"]]
        label_names = ["ham", "spam"]

        ds = {"data": x_train, "target": y_train, "names": label_names}
        return preprocess_and_noise(dataset=ds, mu=mu)

    return closure


def load_pulsars():
    """
    Load pulsars dataset. This is done using scipy#s method to load arff files.
    :return:            closure, A closure function providing the dataset on call.
    """
    # Load the dataset if necessary
    maybe_download_dataset(file_name="../res/pulsars/pulsar_stars.arff",
                           url="https://archive.ics.uci.edu/ml/machine-learning-databases/00372/",
                           url_file_name="HTRU2.zip",
                           extract_file='HTRU_2.arff')

    # Define the closure function
    def closure(mu):
        target = 'class'
        data = arff.loadarff("../res/pulsars/pulsar_stars.arff")
        df = pd.DataFrame(data=data[0])
        df = df.dropna(axis=0)

        y_train = LabelEncoder().fit_transform(df[target])
        X_train = df.drop(target, axis=1)
        ds = {"data": X_train, "target": y_train}
        return preprocess_and_noise(dataset=ds, mu=mu)

    return closure


def load_sdss():
    """
    Loads the SDSS data set.
    :return:            closure, A closure function providing the dataset on call.
    """
    # Define the closure function
    def closure(mu):
        df = pd.read_csv("../res/kaggle/Skyserver_SQL2_27_2018 6_51_39 PM.csv", encoding="utf-8")
        df = df.dropna(axis=0)

        y_train = LabelEncoder().fit_transform(df["class"])
        X_train = df.drop("class", axis=1)
        ds = {"data": X_train, "target": y_train}
        return preprocess_and_noise(dataset=ds, mu=mu)

    return closure


def load_twitter_airline():
    """
    Load datasets originally obtained from kaggle.
    :return:            closure, A closure function providing the dataset on call.
    """
    # Define the closure function
    def closure(mu):
        df = pd.read_csv("../res/kaggle/Tweets.csv", encoding="utf-8", usecols=["airline_sentiment", "text"])
        df = df.dropna(axis=0)

        y_train = LabelEncoder().fit_transform(df["airline_sentiment"])
        X_train = df["text"]
        ds = {"data": X_train, "target": y_train}
        return preprocess_and_noise(dataset=ds, mu=mu)

    return closure
