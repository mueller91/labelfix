import sklearn
from src.labelfix import check_dataset, preprocess_x_y_and_shuffle
from src.utils.view_20_newsgroup import view_twenty

# example on how the system works on textual data
if __name__ == "__main__":
    # load the twenty newsgroup data set
    twenty_newsgroup = sklearn.datasets.fetch_20newsgroups(subset="all", shuffle=False)

    # "data" is required to be a list of strings. Each string is the newsgroup article to be classified.
    # "target" is an array of ints representing the labels.
    twenty_newsgroup["data"], twenty_newsgroup["target"] = preprocess_x_y_and_shuffle(twenty_newsgroup["data"], twenty_newsgroup["target"])
    res = check_dataset(twenty_newsgroup["data"], twenty_newsgroup["target"])

    # return first 100 questionable indices
    print("The first 100 questionable pairs (x_i, y_i) are: {}".format(res["indices"][:100]))

    # iterate over the findings and display both X (from the original corpus) and the questionable labels y
    for i in res["indices"]:
        print("Loading next document .. please be patient\n")
        view_twenty(sklearn.datasets.fetch_20newsgroups(subset="all", shuffle=False), i)
        input("... Press Return for next")
