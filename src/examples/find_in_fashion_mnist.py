from src.utils.visualize import visualize_image
from src.labelfix import check_dataset, preprocess_x_y_and_shuffle
import tensorflow as tf

# In this example, we aim to find mislabeled instances in the fashion MNIST training data set
if __name__ == "__main__":
    # First, construct required dictionary using the fashion mnist training data
    (x_train, y_train), (_, _) = tf.keras.datasets.fashion_mnist.load_data()

    # check the data set
    x_train, y_train = preprocess_x_y_and_shuffle(x_train, y_train)
    res = check_dataset(x_train, y_train,
                        hyperparams={'activation': 'relu', 'dropout': 0.3, 'learn_rate': 0.001, 'num_hidden': 3,
                                     'output_dim': 10, 'input_dim': 2048})

    # plot four sets of images with the most likely mislabeled pairs (x, y) and save to disk
    for i in range(40):
        visualize_image(image_data=x_train,
                        image_labels=y_train,
                        label_names=["top/shirt", "trousers", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"],
                        indices=res["indices"],
                        batch_to_plot=i,
                        save_to_path="../../out/fashion/")

    ids = res["indices"][:int(res["indices"].shape[0] * 0.03)]
    print(f"found {ids.shape} indices, they are: {ids}")

    r = [i for i in ids if i in [3415, 28264, 29599, 37286, 36049, 9059, 18188, 1600, 34381, 22750, 39620, 50494, 38208, 53257]]
    print("Res:")
    print(r)

