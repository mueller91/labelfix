from sklearn.datasets import fetch_20newsgroups


def view_twenty(twenty_data, i):
    """
    Somewhat nicely print an element of the twenty news group data set.
    :param twenty_data:         twenty news group data set as given by sklearns method for loading
    :param i:                   int, index of the element to be displayed
    :return:                    None
    """
    # y
    target_i = twenty_data.target[i]
    category = twenty_data.target_names[target_i]
    print("====== {} ====== (Index {})\n".format(category, i))

    # X
    text = twenty_data.data[i]
    print(text)
    print("\n\n")


if __name__ == "__main__":
    twenty_train = fetch_20newsgroups(subset="all", shuffle=False)
    view_twenty(twenty_train, 13622)
