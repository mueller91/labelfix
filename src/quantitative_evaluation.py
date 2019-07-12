# ================================= Imports =======================================
import datetime

from sklearn.datasets import load_wine, load_breast_cancer, load_iris, load_digits
from tabulate import tabulate

from src.datasets.cifar100_classes import cifar_cls
from src.labelfix import check_dataset, _calc_recall, _precision_on_k
from src.datasets.load_datasets import *


# ================================== SETTINGS =====================================
# Factor of added noise
MU_CONSTANT = 0.03
print("MU={}".format(MU_CONSTANT))

# =================================================================================


def check_dataset_super(current_dataset_fnct, ds_name, mu, df=None):
    """
    Given a DataFrame, dataset and classifier, perform whole pipeline of adding noise and classification.
    :param current_dataset_fnct:    closure, returns a data set.
    :param ds_name:                 str, name of the data set for display
    :param mu:                      float, percentage of data to which label noise will be added, has to be in [0,1]
    :param df:                      DataFrame, results will be appended to this DataFrame, if None, a new one will be created
    :return:                        DataFrame, results are appended appended
    """
    # if necessary create new DataFrame
    if df is None:
        df = pd.DataFrame()

    # get the data set
    current_dataset = current_dataset_fnct(mu)

    # check the dataset
    res = check_dataset(X=current_dataset["data"], y=current_dataset["target_noisy"])

    # calc goodness / recall
    recall = {}
    for alpha in [0.01, 0.02, 0.03]:
        recall[str(alpha)] = _calc_recall(pred=res["pred"],
                                          y_noisy=current_dataset["target_noisy"],
                                          y_true=current_dataset["target"],
                                          alpha=alpha, mu=mu)

    # calc accuracy / precision
    precision = {}
    for alpha in [0.01, 0.02, 0.03]:
        precision[str(alpha)] = _precision_on_k(pred=res["pred"],
                                                y_noisy=current_dataset["target_noisy"],
                                                y_true=current_dataset["target"],
                                                alpha=alpha)
    print("Found recall {} and precision {}".format(recall, precision))

    # ================================================================================================================
    # noinspection PyTypeChecker
    return df.append({"Dataset": ds_name, "Size": current_dataset["data"].shape,
                      "Classes": len(np.unique(current_dataset["target_noisy"], axis=0)),
                      "Runtime": res["runtime"],
                      "rec {}".format(0.01): round(recall["0.01"], 3),
                      "rec {}".format(0.02): round(recall["0.02"], 3),
                      "rec {}".format(0.03): round(recall["0.03"], 3),
                      "prec {}".format(0.01): round(precision["0.01"], 3),
                      "prec {}".format(0.02): round(precision["0.02"], 3),
                      "prec {}".format(0.03): round(precision["0.03"], 3),
                      "target imbalance": round(res["imbalance"], 3),
                      "very best hyperparams": res["best_params"]
                      }, ignore_index=True)


def single_run_over_all(run_number=0):
    """
    Go through all data sets once and calculate metrics, etc.
    :param run_number:      int, the number of the run for identification
    :return:                None
    """
    # Create a DataFrame for storing results
    df = pd.DataFrame()

    # Define all data sets. Comment out the ones, which should not be run.
    all_datasets = [
        # ===================== Bigger and large real-world datasets ======
        (load_toy_cov(), "forest covertype (10%)"),
        (load_ccd(), "credit card default"),
        (load_adult(), "adult"),
        # ==================== Small toy datasets ========================
        (load_toy(load_iris), "iris"),
        (load_toy(load_wine), "wine"),
        (load_toy(load_breast_cancer), "breast_cancer"),
        (load_toy(load_digits), "digits"),
        # ==================== Synthetic datasets ======================
        (synth_classification(n_samples=10000, n_features=9, n_classes=3, n_informative=9, n_redundant=0, n_repeated=0), "synthetic 1"),
        (synth_classification(n_samples=10000, n_features=9, n_classes=5, n_informative=9, n_redundant=0, n_repeated=0), "synthetic 2"),
        (synth_classification(n_samples=10000, n_features=45, n_classes=7, n_informative=45, n_redundant=0, n_repeated=0), "synthetic 3"),
        (synth_classification(n_samples=10000, n_features=45, n_classes=15, n_informative=45, n_redundant=0, n_repeated=0), "synthetic 4"),
        (synth_classification(n_samples=10000, n_features=85, n_classes=7, n_informative=85, n_redundant=0, n_repeated=0), "synthetic 5"),
        (synth_classification(n_samples=10000, n_features=85, n_classes=15, n_informative=85, n_redundant=0, n_repeated=0), "synthetic 5"),
        # ============== BLOB ============
        (synth_classification(n_samples=4000, n_features=12, n_classes=12, n_informative=9, n_redundant=1, n_repeated=1, use_blob=True), "synthetic blobs"),
        # # ======================= KAGGLE ==============================
        (load_pulsars(), "pulsar_stars"),
        # Note that SDSS dataset has to be downloaded from kaggle manually https://www.kaggle.com/lucidlenn/sloan-digital-sky-survey
        # (load_sdss(), "sloan-digital-sky-survey"),
        # ====================== Natural language =========================
        # Note that the twitter airline dataset has to be downloaded from kaggle manually https://www.kaggle.com/crowdflower/twitter-airline-sentiment
        # (load_twitter_airline(), "twitter airline"),
        (load_sms_spam(), "sms spam"),
        (load_imdb(), "imdb"),
        (load_twenty_newsgroup(), "twenty newsgroup"),
        # ===================== Image datasets ============================
        (load_svhn(), "svhn"),
        (load_mnist(), "mnist"),
        (load_fashion_mnist(), "fashion-mnist"),
        (load_cifar(num=10), "cifar10"),
        (load_cifar(num=100), "cifar100"),
        (load_cifar(num=100, subsection=cifar_cls['aquatic_mammals']), "cifar100, subset aqua"),
        (load_cifar(num=100, subsection=cifar_cls['flowers']), "cifar100, subset flowers"),
        (load_cifar(num=100, subsection=cifar_cls['household_electrical_devices']), "cifar100, subset household"),
        (load_cifar(num=100, switch_dict=cifar_cls), "cifar100, at random"),

    ]

    # iterate over data sets and calculate metrics
    for dataset_func, dataset_name in all_datasets:
        print("\n== >> Processing data set {}".format(dataset_name))
        df = check_dataset_super(dataset_func, dataset_name, df=df, mu=MU_CONSTANT)
        print(tabulate(df, headers='keys', tablefmt='psql'))

    # Write to latex
    latex_path = "../doc/{}run{}.tex".format(datetime.datetime.now(), run_number)
    print("Writing results as latex table to {}".format(latex_path))
    if not os.path.exists(os.path.dirname(latex_path)):
        os.makedirs(os.path.dirname(latex_path))

    with open(latex_path, "w") as f:
        df = df.sort_values("Dataset")
        df.to_latex(f, index=False, bold_rows=True)

    df.to_csv(latex_path.replace("tex", "csv"), sep="\t")

    # aggregate
    df_agg = df.copy(True)
    df_agg.drop("Size", axis=1, inplace=True)
    df_agg.drop("Runtime", axis=1, inplace=True)
    df_agg.drop("Dataset", axis=1, inplace=True)
    df_agg = df_agg.mean(axis=0)
    print(tabulate(df, headers='keys', tablefmt='psql'))

    df_agg.to_csv(latex_path.replace(".tex", ".agg.csv"), sep="\t")


if __name__ == "__main__":
    for i in range(5):
        print("RUN " + str(i))
        single_run_over_all(i)
