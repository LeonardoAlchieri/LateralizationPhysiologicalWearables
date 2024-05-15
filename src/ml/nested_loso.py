from typing import Callable
from itertools import product

import pandas as pd
from joblib import Parallel, delayed
from numpy import hstack, ndarray
from numpy.random import randint
from numpy.random import seed as set_numpy_seed
from pandas import DataFrame, IndexSlice, concat
from tqdm.auto import tqdm
from joblib_progress import joblib_progress

from src.ml.nested import (
    fit_with_hyperparameters,
    resample_train_data,
)


def train_score_single_model(
    groups: ndarray,
    x: ndarray,
    y: ndarray,
    user: str,
    random_state_classifier: int,
    random_state_undersampling: int,
    random_state_fold: int,
    n_inner_folds: int,
    resampling_method: Callable | None = None,
    **kwargs,
) -> DataFrame:
    """
    Train and score a single model using a lazy classifier and return a DataFrame with the model's scores.

    Parameters
    ----------
    groups_train :
        The groups in the training set.
    groups_test :
        The groups in the testing set.
    x_train : ndarray
        The features in the training set.
    x_test : ndarray
        The features in the testing set.
    y_train : ndarray
        The target values in the training set.
    y_test : ndarray
        The target values in the testing set.
    ml_model : LazyClassifier
        The lazy classifier to use for training and scoring the model.
    classifier_seed : int
        The random seed to use for the classifier.
    user : str
        The user for whom to train and score the model.

    Returns
    -------
    DataFrame
        A DataFrame with the model's scores.
    """
    train_data_mask: ndarray = groups != user
    x_train: ndarray = x[train_data_mask]
    groups_train: ndarray = groups[train_data_mask]

    test_data_mark: ndarray = groups == user
    x_test: ndarray = x[test_data_mark]
    groups_test: ndarray = groups[test_data_mark]

    y_train: ndarray = y[train_data_mask]
    y_test: ndarray = y[test_data_mark]

    x_resampled_train, y_resampled_train = resample_train_data(
        x_train=x_train,
        y_train=y_train,
        groups_train=groups_train,
        resampling_method=resampling_method,
        random_state_undersampling=random_state_undersampling,
    )

    models = fit_with_hyperparameters(
        x_train=x_resampled_train,
        y_train=y_resampled_train,
        x_test=x_test,
        y_test=y_test,
        random_state_classifier=random_state_classifier,
        random_state_fold=random_state_fold,
        n_inner_folds=n_inner_folds,
    )
    models.columns.name = user
    return models


def train_score_single_model_opposite_side(
    users_train: ndarray,
    users_test: ndarray,
    features_train: ndarray,
    labels_train: ndarray,
    features_test: ndarray,
    labels_test: ndarray,
    user: str,
    random_state_classifier: int,
    random_state_undersampling: int,
    random_state_fold: int,
    n_inner_folds: int,
    resampling_method: Callable | None = None,
    **kwargs,
) -> DataFrame:
    """
    Train and score a single model using a lazy classifier and return a DataFrame with the model's scores.

    Parameters
    ----------
    users_train :
        The groups in the training set.
    groups_test :
        The groups in the testing set.
    x_train : ndarray
        The features in the training set.
    x_test : ndarray
        The features in the testing set.
    y_train : ndarray
        The target values in the training set.
    y_test : ndarray
        The target values in the testing set.
    ml_model : LazyClassifier
        The lazy classifier to use for training and scoring the model.
    classifier_seed : int
        The random seed to use for the classifier.
    user : str
        The user for whom to train and score the model.

    Returns
    -------
    DataFrame
        A DataFrame with the model's scores.
    """
    train_data_mask: ndarray = users_train != user
    x_train: ndarray = features_train[train_data_mask]
    groups_train: ndarray = users_train[train_data_mask]

    test_data_mark: ndarray = users_test == user
    x_test: ndarray = features_test[test_data_mark]
    groups_test: ndarray = users_test[train_data_mask]

    y_train: ndarray = labels_train[train_data_mask]
    y_test: ndarray = labels_test[test_data_mark]

    x_resampled_train, y_resampled_train = resample_train_data(
        x_train=x_train,
        y_train=y_train,
        groups_train=groups_train,
        resampling_method=resampling_method,
        random_state_undersampling=random_state_undersampling,
    )

    models = fit_with_hyperparameters(
        x_train=x_resampled_train,
        y_train=y_resampled_train,
        x_test=x_test,
        y_test=y_test,
        random_state_classifier=random_state_classifier,
        random_state_fold=random_state_fold,
        n_inner_folds=n_inner_folds,
    )
    models.columns.name = user
    return models


def LOSO(
    groups: ndarray,
    x: ndarray,
    y: ndarray,
    random_state_classifier: int,
    random_state_undersampling: int,
    random_state_fold: int,
    n_inner_folds: int,
    resampling_method: Callable | None = None,
) -> list[DataFrame]:
    """
    Perform Leave-One-Subject-Out (LOSO) cross-validation for training and scoring a lazy classifier, and return a list
    of DataFrames with the scores for each user.

    Parameters
    ----------
    groups_train : list[str]
        The groups in the training set.
    groups_test : list[str]
        The groups in the testing set.
    x_train : ndarray
        The features in the training set.
    x_test : ndarray
        The features in the testing set.
    y_train : ndarray
        The target values in the training set.
    y_test : ndarray
        The target values in the testing set.
    ml_model : LazyClassifier
        The lazy classifier to use for training and scoring the model.
    classifier_seed : int, optional
        The random seed to use for the classifier. Default is 42.
    n_jobs : int, optional
        The number of jobs to run in parallel. Default is 1.
    augment_data : bool, optional
        Whether to augment the data or not. Default is False.

    Returns
    -------
    list
        A list of DataFrames with the scores for each user.
    """
    scores = [
        train_score_single_model(
            groups=groups,
            x=x,
            y=y,
            user=user,
            random_state_classifier=random_state_classifier,
            random_state_undersampling=random_state_undersampling,
            random_state_fold=random_state_fold,
            n_inner_folds=n_inner_folds,
            resampling_method=resampling_method,
        )
        for user in set(groups)
    ]
    return scores


def compute_loso_same_side(
    data: DataFrame,
    random_state_classifier: int,
    random_state_undersampling: int,
    random_state_fold: int,
    n_inner_folds: int,
    resampling_method: Callable | None = None,
):
    x = data.drop(columns=["label"], inplace=False).values
    y = data["label"].values
    groups = data.index.get_level_values(0).values

    all_models = LOSO(
        groups=groups,
        x=x,
        y=y,
        random_state_classifier=random_state_classifier,
        random_state_undersampling=random_state_undersampling,
        random_state_fold=random_state_fold,
        n_inner_folds=n_inner_folds,
        resampling_method=resampling_method,
    )

    averages = (
        concat(all_models)
        .groupby(level=0)
        .mean()
        .sort_values(by="Balanced Accuracy", ascending=False)
    )
    standard_deviations = (
        concat(all_models)
        .groupby(level=0)
        .std()
        .sort_values(by="Balanced Accuracy", ascending=False)
    )
    standard_errors = standard_deviations / (len(all_models) ** 0.5)

    return all_models, concat(
        [averages, standard_errors],
        axis=1,
        keys=["Average", "Standard error"],
    )


def run_same_side_classifications(
    x: ndarray,
    y: ndarray,
    folds: ndarray,
    generator_seeds: tuple[int, int] = (42, 69),
    n_seeds_to_undersample: int = 10,
    n_seeds_to_test_classifiers: int = 10,
    n_seeds_to_test_folds: int = 10,
    n_inner_folds: int = 5,
    **kwargs,
) -> tuple[DataFrame, list[list[DataFrame]]]:
    """
    The run_same_side_classifications function takes in a set of feature
    vectors x and their corresponding labels y, along with a set of
    cross-validation folds folds. The function resamples the data using
    the resampling function and trains a LazyClassifier model using the
    LOSO method, which performs leave-one-subject-out cross-validation.
    The function repeats this process for a set of different random seeds
    for the classifier, computing the average accuracy and standard error
    across all runs for each classifier seed.

    Parameters:

    x : ndarray
    The feature vectors, of shape (n_samples, n_features)
    y : ndarray
    The corresponding labels for the feature vectors, of shape (n_samples,)
    folds : ndarray
    The cross-validation fold assignments for each sample, of shape (n_samples,)
    Returns:

    DataFrame
    A pandas DataFrame with the average accuracy and standard error for
    each classifier seed, sorted by accuracy in descending order.
    The DataFrame has two levels of columns, with the top level
    being "Average" and "Standard error", and the bottom level being
    "Balanced Accuracy", "ROC AUC", "F1 score", and "Balanced accuracy".
    """

    # NOTE: we still set a single seed, from which we generate a bunch of other
    # random seeds to be fed to the algorithm
    set_numpy_seed(generator_seeds[0])
    random_states_classifiers = randint(
        0, int(2**32 - 1), n_seeds_to_test_classifiers
    )

    # NOTE: to avoid dependencies between the seeds for the classifiers and those
    # for the cross validation, two "main" seeds are required, from which then
    # generate all of the others. This also allows reproducibility of the code.
    set_numpy_seed(generator_seeds[1])
    random_states_folds = randint(0, int(2**32 - 1), n_seeds_to_test_folds)

    set_numpy_seed(generator_seeds[2])
    random_states_undersampling = randint(0, int(2**32 - 1), n_seeds_to_undersample)

    x = x.reshape((x.shape[0], -1))
    data = DataFrame(x, index=folds)
    data["label"] = y

    results = []
    all_results: list[list[DataFrame]] = []

    possible_combinations = list(
        product(
            random_states_folds, random_states_undersampling, random_states_classifiers
        )
    )

    with joblib_progress(
        "Random seed iterations", total=len(list(possible_combinations))
    ):
        outer_folds_output: list[tuple[list, list[list[DataFrame]]]] = Parallel(
            n_jobs=kwargs.get("n_jobs", 1), backend="loky"
        )(
            delayed(compute_loso_same_side)(
                data=data,
                random_state_classifier=random_state_classifier,
                random_state_undersampling=random_state_undersampling,
                random_state_fold=random_state_fold,
                n_inner_folds=n_inner_folds,
                resampling_method=kwargs.get("resampling_method", None),
            )
            for random_state_fold, random_state_undersampling, random_state_classifier in possible_combinations
        )

    results = [outer_fold[1] for outer_fold in outer_folds_output]
    all_results = [outer_fold[0] for outer_fold in outer_folds_output]

    averages_seeds = (
        concat(results)
        .groupby(level=0)
        .apply(lambda x: x.loc[:, IndexSlice["Average", :]].mean())
        .droplevel(axis=1, level=0)
        .sort_values(by=("Balanced Accuracy"), ascending=False)
    )

    errors_seeds = (
        concat(results)
        .groupby(level=0)
        .apply(
            lambda x: (x.loc[:, IndexSlice["Standard error", :]] ** 2).sum() ** 0.5
            / (
                n_seeds_to_test_classifiers
                * n_seeds_to_undersample
                * n_seeds_to_test_folds
            )
        )
        .droplevel(axis=1, level=0)
        .sort_values(by="Balanced Accuracy", ascending=False)
    )
    return (
        concat(
            [averages_seeds, errors_seeds], axis=1, keys=["Average", "Standard error"]
        ),
        all_results,
    )


# def under_sampling(
#     x: ndarray,
#     y: ndarray,
#     folds: ndarray,
#     resampling_method: Callable,
#     random_state: int = 42,
# ) -> tuple[ndarray, ndarray, ndarray]:
#     """
#     Perform under-sampling using the given resampling method for each fold in the data.

#     Parameters
#     ----------
#     x : ndarray
#         The features of the data.
#     y : ndarray
#         The target labels of the data.
#     folds : ndarray
#         The array indicating which fold each sample belongs to.
#     resampling_method : Callable
#         The callable object used to resample the data for each fold.
#     random_state : int, optional
#         The seed used for the resampling, by default 42.

#     Returns
#     -------
#     tuple[ndarray, ndarray, ndarray]
#         A tuple containing the resampled features, target labels, and fold labels.
#     """
#     x = x.reshape((x.shape[0], -1))
#     data = DataFrame(x, index=folds)
#     data["label"] = y
#     data_resampled = data.groupby(axis=0, level=0).apply(
#         resampling, resampling_method=resampling_method, random_state=random_state
#     )
#     x_resampled = data_resampled.drop(columns=["label"], inplace=False).values
#     y_resampled = data_resampled["label"].values
#     folds_resampled = data_resampled.index.get_level_values(0).values

#     return x_resampled, y_resampled, folds_resampled


def LOSO_opposite_side(
    users_train: ndarray,
    users_test: ndarray,
    features_train: ndarray,
    labels_train: ndarray,
    features_test: ndarray,
    labels_test: ndarray,
    random_state_classifier: int,
    random_state_undersampling: int,
    random_state_fold: int,
    n_inner_folds: int,
    resampling_method: Callable | None = None,
):
    scores = [
        train_score_single_model_opposite_side(
            users_train=users_train,
            users_test=users_test,
            features_train=features_train,
            labels_train=labels_train,
            features_test=features_test,
            labels_test=labels_test,
            user=user,
            random_state_classifier=random_state_classifier,
            random_state_undersampling=random_state_undersampling,
            random_state_fold=random_state_fold,
            n_inner_folds=n_inner_folds,
            resampling_method=resampling_method,
        )
        for user in set(users_train)
    ]
    return scores


def compute_loso_opposite_side(
    features_right: ndarray,
    labels_right: ndarray,
    groups_right: ndarray,
    features_left: ndarray,
    labels_left: ndarray,
    groups_left: ndarray,
    random_state_undersampling: int,
    random_state_classifier: int,
    random_state_fold: int,
    n_inner_folds: int,
    which_comparison: str,
    resampling_method: Callable,
    **kwargs,
) -> tuple[DataFrame, list[list[DataFrame]]]:
    """
    Run multiple iterations of different classifiers on the resampled training and testing data,
    and return a DataFrame with the average accuracy and standard error of the different models.

    Parameters
    ----------
    x_train : ndarray
        The features of the training data.
    x_test : ndarray
        The features of the testing data.
    y_train : ndarray
        The target labels of the training data.
    y_test : ndarray
        The target labels of the testing data.
    folds_train : int
        The number of folds to generate for the training data.
    folds_test : int
        The number of folds to generate for the testing data.
    n_jobs : int
        The number of CPU cores to use for parallel processing.

    Returns
    -------
    DataFrame
        A DataFrame containing the average accuracy and standard error of the different models.
    """

    if which_comparison == "rxlx":
        all_models = LOSO_opposite_side(
            users_train=groups_right,
            users_test=groups_left,
            features_train=features_right,
            labels_train=labels_right,
            features_test=features_left,
            labels_test=labels_left,
            random_state_classifier=random_state_classifier,
            random_state_undersampling=random_state_undersampling,
            random_state_fold=random_state_fold,
            n_inner_folds=n_inner_folds,
            resampling_method=resampling_method,
        )
    elif which_comparison == "lxrx":
        all_models = LOSO_opposite_side(
            users_train=groups_left,
            users_test=groups_right,
            features_train=features_left,
            labels_train=labels_left,
            features_test=features_right,
            labels_test=labels_right,
            random_state_classifier=random_state_classifier,
            random_state_undersampling=random_state_undersampling,
            random_state_fold=random_state_fold,
            n_inner_folds=n_inner_folds,
            resampling_method=resampling_method,
        )
    else:
        raise ValueError(
            f"which_comparison must be either 'rxlx' or 'lxrx'. Received {which_comparison}"
        )

    averages = (
        concat(all_models)
        .groupby(level=0)
        .mean()
        .sort_values(by="Balanced Accuracy", ascending=False)
    )
    standard_deviations = (
        concat(all_models)
        .groupby(level=0)
        .std()
        .sort_values(by="Balanced Accuracy", ascending=False)
    )
    standard_errors = standard_deviations / (len(all_models) ** 0.5)

    return all_models, concat(
        [averages, standard_errors],
        axis=1,
        keys=["Average", "Standard error"],
    )


def run_opposite_side_prediction(
    features_right: ndarray,
    labels_right: ndarray,
    groups_right: ndarray,
    features_left: ndarray,
    labels_left: ndarray,
    groups_left: ndarray,
    which_comparison: str,
    generator_seeds: tuple[int, int] = [42, 69, 666],
    n_seeds_to_test_classifiers: int = 10,
    n_seeds_to_undersample: int = 10,
    n_seeds_to_test_folds: int = 10,
    n_inner_folds: int = 5,
    **kwargs,
) -> tuple[DataFrame, list[list[DataFrame]]]:
    """
    Run multiple iterations of different classifiers on the resampled training and testing data,
    and return a DataFrame with the average accuracy and standard error of the different models.

    Parameters
    ----------
    x_train : ndarray
        The features of the training data.
    x_test : ndarray
        The features of the testing data.
    y_train : ndarray
        The target labels of the training data.
    y_test : ndarray
        The target labels of the testing data.
    folds_train : int
        The number of folds to generate for the training data.
    folds_test : int
        The number of folds to generate for the testing data.
    n_jobs : int
        The number of CPU cores to use for parallel processing.

    Returns
    -------
    DataFrame
        A DataFrame containing the average accuracy and standard error of the different models.
    """
    # NOTE: we still set a single seed, from which we generate a bunch of other
    # random seeds to be fed to the algorithm
    set_numpy_seed(generator_seeds[0])
    random_states_classifiers = randint(
        0, int(2**32 - 1), n_seeds_to_test_classifiers
    )

    # NOTE: to avoid dependencies between the seeds for the classifiers and those
    # for the cross validation, two "main" seeds are required, from which then
    # generate all of the others. This also allows reproducibility of the code.
    set_numpy_seed(generator_seeds[1])
    random_states_folds = randint(0, int(2**32 - 1), n_seeds_to_test_folds)

    set_numpy_seed(generator_seeds[2])
    random_states_undersampling = randint(0, int(2**32 - 1), n_seeds_to_undersample)

    results: list[DataFrame]
    all_results: list[list[DataFrame]]

    possible_combinations = list(
        product(
            random_states_folds, random_states_undersampling, random_states_classifiers
        )
    )
    with joblib_progress(
        "Random seed iterations", total=len(list(possible_combinations))
    ):
        outer_folds_output: list[tuple[list, list[list[DataFrame]]]] = Parallel(
            n_jobs=kwargs.get("n_jobs", 1),
        )(
            delayed(compute_loso_opposite_side)(
                features_right=features_right.reshape((features_right.shape[0], -1)),
                labels_right=labels_right.reshape((labels_right.shape[0], -1)),
                groups_right=groups_right,
                features_left=features_left.reshape((features_left.shape[0], -1)),
                labels_left=labels_left.reshape((labels_left.shape[0], -1)),
                groups_left=groups_left,
                random_state_undersampling=random_state_undersampling,
                random_state_classifier=random_state_classifier,
                random_state_fold=random_state_fold,
                n_inner_folds=n_inner_folds,
                which_comparison=which_comparison,
                resampling_method=kwargs.get("resampling_method", None),
            )
            for random_state_fold, random_state_undersampling, random_state_classifier in possible_combinations
        )

    results = [outer_fold[1] for outer_fold in outer_folds_output]
    all_results = [outer_fold[0] for outer_fold in outer_folds_output]

    averages_seeds = (
        pd.concat(results)
        .groupby(level=0)
        .apply(lambda x: x.loc[:, IndexSlice["Average", :]].mean())
        .droplevel(axis=1, level=0)
        .sort_values(by=("Balanced Accuracy"), ascending=False)
    )

    errors_seeds = (
        pd.concat(results)
        .groupby(level=0)
        .apply(
            lambda x: (x.loc[:, IndexSlice["Standard error", :]] ** 2).sum() ** 0.5
            / (n_seeds_to_test_classifiers * n_seeds_to_undersample)
        )
        .droplevel(axis=1, level=0)
        .sort_values(by="Balanced Accuracy", ascending=False)
    )
    return (
        pd.concat(
            [averages_seeds, errors_seeds], axis=1, keys=["Average", "Standard error"]
        ),
        all_results,
    )

def compute_loso_opposite_side_updated(
    features_train: ndarray,
    labels_train: ndarray,
    groups_train: ndarray,
    features_test: ndarray,
    labels_test: ndarray,
    groups_test: ndarray,
    random_state_undersampling: int,
    random_state_classifier: int,
    random_state_fold: int,
    n_inner_folds: int,
    resampling_method: Callable,
    **kwargs,
) -> tuple[DataFrame, list[list[DataFrame]]]:
    """
    Run multiple iterations of different classifiers on the resampled training and testing data,
    and return a DataFrame with the average accuracy and standard error of the different models.

    Parameters
    ----------
    x_train : ndarray
        The features of the training data.
    x_test : ndarray
        The features of the testing data.
    y_train : ndarray
        The target labels of the training data.
    y_test : ndarray
        The target labels of the testing data.
    folds_train : int
        The number of folds to generate for the training data.
    folds_test : int
        The number of folds to generate for the testing data.
    n_jobs : int
        The number of CPU cores to use for parallel processing.

    Returns
    -------
    DataFrame
        A DataFrame containing the average accuracy and standard error of the different models.
    """

    all_models = LOSO_opposite_side(
            users_train=groups_train,
            users_test=groups_test,
            features_train=features_train,
            labels_train=labels_train,
            features_test=features_test,
            labels_test=labels_test,
            random_state_classifier=random_state_classifier,
            random_state_undersampling=random_state_undersampling,
            random_state_fold=random_state_fold,
            n_inner_folds=n_inner_folds,
            resampling_method=resampling_method,
        )

    averages = (
        concat(all_models)
        .groupby(level=0)
        .mean()
        .sort_values(by="Balanced Accuracy", ascending=False)
    )
    standard_deviations = (
        concat(all_models)
        .groupby(level=0)
        .std()
        .sort_values(by="Balanced Accuracy", ascending=False)
    )
    standard_errors = standard_deviations / (len(all_models) ** 0.5)

    return all_models, concat(
        [averages, standard_errors],
        axis=1,
        keys=["Average", "Standard error"],
    )
    
def run_opposite_side_prediction_updated(
    features_train: ndarray,
    labels_train: ndarray,
    groups_train: ndarray,
    features_test: ndarray,
    labels_test: ndarray,
    groups_test: ndarray,
    generator_seeds: tuple[int, int] = [42, 69, 666],
    n_seeds_to_test_classifiers: int = 10,
    n_seeds_to_undersample: int = 10,
    n_seeds_to_test_folds: int = 10,
    n_inner_folds: int = 5,
    **kwargs,
) -> tuple[DataFrame, list[list[DataFrame]]]:
    """
    Run multiple iterations of different classifiers on the resampled training and testing data,
    and return a DataFrame with the average accuracy and standard error of the different models.

    Parameters
    ----------
    x_train : ndarray
        The features of the training data.
    x_test : ndarray
        The features of the testing data.
    y_train : ndarray
        The target labels of the training data.
    y_test : ndarray
        The target labels of the testing data.
    folds_train : int
        The number of folds to generate for the training data.
    folds_test : int
        The number of folds to generate for the testing data.
    n_jobs : int
        The number of CPU cores to use for parallel processing.

    Returns
    -------
    DataFrame
        A DataFrame containing the average accuracy and standard error of the different models.
    """
    # NOTE: we still set a single seed, from which we generate a bunch of other
    # random seeds to be fed to the algorithm
    set_numpy_seed(generator_seeds[0])
    random_states_classifiers = randint(
        0, int(2**32 - 1), n_seeds_to_test_classifiers
    )

    # NOTE: to avoid dependencies between the seeds for the classifiers and those
    # for the cross validation, two "main" seeds are required, from which then
    # generate all of the others. This also allows reproducibility of the code.
    set_numpy_seed(generator_seeds[1])
    random_states_folds = randint(0, int(2**32 - 1), n_seeds_to_test_folds)

    set_numpy_seed(generator_seeds[2])
    random_states_undersampling = randint(0, int(2**32 - 1), n_seeds_to_undersample)

    results: list[DataFrame]
    all_results: list[list[DataFrame]]

    possible_combinations = list(
        product(
            random_states_folds, random_states_undersampling, random_states_classifiers
        )
    )
    with joblib_progress(
        "Random seed iterations", total=len(list(possible_combinations))
    ):
        outer_folds_output: list[tuple[list, list[list[DataFrame]]]] = Parallel(
            n_jobs=kwargs.get("n_jobs", 1),
        )(
            delayed(compute_loso_opposite_side_updated)(
                features_train=features_train,
                labels_train=labels_train,
                groups_train=groups_train,
                features_test=features_test,
                labels_test=labels_test,
                groups_test=groups_test,
                random_state_undersampling=random_state_undersampling,
                random_state_classifier=random_state_classifier,
                random_state_fold=random_state_fold,
                n_inner_folds=n_inner_folds,
                resampling_method=kwargs.get("resampling_method", None),
            )
            for random_state_fold, random_state_undersampling, random_state_classifier in possible_combinations
        )

    results = [outer_fold[1] for outer_fold in outer_folds_output]
    all_results = [outer_fold[0] for outer_fold in outer_folds_output]

    averages_seeds = (
        pd.concat(results)
        .groupby(level=0)
        .apply(lambda x: x.loc[:, IndexSlice["Average", :]].mean())
        .droplevel(axis=1, level=0)
        .sort_values(by=("Balanced Accuracy"), ascending=False)
    )

    errors_seeds = (
        pd.concat(results)
        .groupby(level=0)
        .apply(
            lambda x: (x.loc[:, IndexSlice["Standard error", :]] ** 2).sum() ** 0.5
            / (n_seeds_to_test_classifiers * n_seeds_to_undersample)
        )
        .droplevel(axis=1, level=0)
        .sort_values(by="Balanced Accuracy", ascending=False)
    )
    return (
        pd.concat(
            [averages_seeds, errors_seeds], axis=1, keys=["Average", "Standard error"]
        ),
        all_results,
    )