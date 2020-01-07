import pickle

from keras.datasets import cifar100

from src.utils.visualize import visualize_image
from src.labelfix import check_dataset, preprocess_x_y_and_shuffle

# In this example, we aim to find mislabeled instances in the CIFAR-100 training data set
if __name__ == "__main__":
    # First, construct required dictionary using the CIFAR-100 training data
    (x_train, y_train), (_, _) = cifar100.load_data(label_mode='fine')

    x_train, y_train = preprocess_x_y_and_shuffle(x_train, y_train)
    res = check_dataset(x_train, y_train)

    # load label names
    with open("../../res/cifar100/cifar100_names", 'rb') as f:
        dic = pickle.load(f, encoding='bytes')
    label_names = dic[b"fine_label_names"]

    # plot four sets of images with the most likely mislabeled pairs (x, y) and save to disk
    for i in range(40):
        visualize_image(image_data=x_train,
                        image_labels=y_train,
                        label_names=label_names,
                        indices=res["indices"],
                        batch_to_plot=i,
                        save_to_path="../../out/cifar100")

    ids = res["indices"][:int(res["indices"].shape[0] * 0.03)]
    print(ids)
    r = [i for i in ids if i in [6093, 24900,33823, 31377, 48760, 31467, 45694]]
    print("Res:")
    print(r)
