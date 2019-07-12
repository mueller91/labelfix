import copy
import os

import matplotlib.pyplot as plt
import numpy as np


def visualize_image(image_data, image_labels, label_names, indices, batch_to_plot=0,
                    additional_str="", save_to_path=None, model_predictions=None):
    """
    Visualize a set of images
    :param image_data:          array of image data, not truncated
    :param image_labels:        array of image labels, not truncated
    :param label_names:         array of image label names, not truncated
    :param indices:             array of int, indices of images sorted by likelihood of being mislabeled
    :param batch_to_plot        int, which batch of indices to plot
    :param additional_str:      string, optional: to add to the output image
    :param model_predictions:   string, optional: Supply the predictions of the retrained model
    :param save_to_path:        string, optional: path where to save images to
    :return:                    None
    """

    image_labels = image_labels.squeeze()
    assert image_data.ndim >= 2, "Invalid dimensionality {} for image data".format(image_data.ndim)
    assert len(image_labels.shape) in [1, 2], "Invalid dimensionality {} for labels".format(len(image_labels.shape))
    assert len(indices.shape) == 1, "Invalid dimensionality {} for indices".format(len(indices.shape))
    assert len(np.unique(label_names)) == len(np.unique(image_labels, axis=0)), \
        "Got {} different label names, but {} different labels. Their ordinarily must be equal!".format(
            len(np.unique(label_names)), len(np.unique(image_labels, axis=0))
        )
    assert isinstance(batch_to_plot, int) and batch_to_plot >= 0, "batch_to_plot must be an integer >= 0"

    indices = indices[batch_to_plot*9:(batch_to_plot+1)*9]
    columns = 3
    fig = plt.figure(figsize=(10, 10))

    X = copy.deepcopy(image_data)
    if X.shape[-1] == 1:
        x_shape_1 = X.shape[:-1]
        X = X.reshape(x_shape_1)

    for i, index in enumerate(indices):
        # Leaves first row empty, since starting at position columns+1
        ax = fig.add_subplot(len(indices) / columns + 1, columns, columns + i + 1)
        ax.imshow(X[index], interpolation='bicubic')

        pred_model = "\nPred: " + str(label_names[model_predictions[index]]) if model_predictions is not None else ""

        try:
            plt.title(label_names[image_labels[index]].decode('ascii') + ", index " + str(index)
                      + pred_model, fontsize=15)
        except:
            plt.title(label_names[image_labels[index]] + ", index " + str(index)
                      + pred_model, fontsize=15)
    fig.suptitle(additional_str + "\n", size=20)
    fig.tight_layout()

    if save_to_path:
        if not os.path.isdir(save_to_path):
            os.makedirs(save_to_path)
        path = os.path.join(save_to_path, "{}.png".format(batch_to_plot))
        plt.savefig(path)
        print("Figure saved to {}".format(path))
    plt.show()
