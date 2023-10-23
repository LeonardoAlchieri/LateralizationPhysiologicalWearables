from typing import Callable

import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from joblib import Parallel, delayed
from lazypredict.Supervised import LazyClassifier
from numpy import hstack, ndarray
from numpy.random import randint
from numpy.random import seed as set_numpy_seed
from pandas import DataFrame, IndexSlice, concat
from tqdm.auto import tqdm

from src.ml import resampling, data_augmentation

def train_score_single_model(
    groups_train: ndarray[str],
    groups_test: ndarray[str],
    x_train: ndarray,
    x_test: ndarray,
    y_train: ndarray,
    y_test: ndarray,
    ml_model: LazyClassifier,
    classifier_seed: int,
    user: str,
    augment_data: bool = False,
    **kwargs,
) -> DataFrame:
    """
    Train and score a single model using a lazy classifier and return a DataFrame with the model's scores.

    Parameters
    ----------
    groups_train : ndarray[str]
        The groups in the training set.
    groups_test : ndarray[str]
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
    augment_data : bool, optional
        Whether to augment the data, by default False.

    Returns
    -------
    DataFrame
        A DataFrame with the model's scores.
    """
    train_data_mask: ndarray = groups_train != user
    train_data: ndarray = x_train[train_data_mask]
    test_data_mark: ndarray = groups_test == user
    test_data: ndarray = x_test[test_data_mark]

    train_labels: ndarray = y_train[train_data_mask]
    test_labels: ndarray = y_test[test_data_mark]
    clf = ml_model(
        predictions=True,
        random_state=classifier_seed,
        classifiers=kwargs.get("classifiers", "all"),
    )

    if augment_data:
        augmentation_params = kwargs.get("augmentation_params", {})
        train_data, train_labels = data_augmentation(
            x=train_data,
            y=train_labels,
            methods=augmentation_params.get(
                "augmentation_methods", ["jitter", "scaling", "flipping", "warping"]
            ),
            aug_percent=augmentation_params.get("augmentation_percent", 0.2),
            seed=augmentation_params.get("augmentation_seed", None),
        )

    models: DataFrame
    models, _ = clf.fit(
        X_train=train_data,
        X_test=test_data,
        y_train=train_labels,
        y_test=test_labels,
    )
    models.columns.name = user
    return models

def LOSO(
    groups_train: list[str],
    groups_test: list[str],
    x_train: ndarray,
    x_test: ndarray,
    y_train: ndarray,
    y_test: ndarray,
    ml_model: LazyClassifier,
    classifier_seed: int = 42,
    n_jobs: int = 1,
    augment_data: bool = False,
    **kwargs,
):
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
    custom_method: Callable | None = kwargs.get("custom_fold_run_method", None)
    method_to_train: Callable
    if custom_method is None:
        method_to_train = train_score_single_model
    else:
        # NOTE: the custom method should have a specific input
        method_to_train = custom_method

    if n_jobs == 1:
        scores = [
            method_to_train(
                groups_train=groups_train,
                groups_test=groups_test,
                x_train=x_train,
                x_test=x_test,
                y_train=y_train,
                y_test=y_test,
                ml_model=ml_model,
                classifier_seed=classifier_seed,
                user=user,
                classifiers=kwargs.get("classifiers", "all"),
                augment_data=augment_data,
                augmentation_params=kwargs.get("augmentation_params", None),
            )
            for user in set(groups_train)
        ]
    else:
        scores = Parallel(n_jobs=n_jobs)(
            delayed(method_to_train)(
                groups_train=groups_train,
                groups_test=groups_test,
                x_train=x_train,
                x_test=x_test,
                y_train=y_train,
                y_test=y_test,
                ml_model=ml_model,
                classifier_seed=classifier_seed,
                user=user,
                classifiers=kwargs.get("classifiers", "all"),
                augment_data=augment_data,
                augmentation_params=kwargs.get("augmentation_params", None),
            )
            for user in set(groups_train)
        )
    return scores


def run_same_side_classifications(
    x: ndarray,
    y: ndarray,
    folds: ndarray,
    generator_seeds: tuple[int, int] = (42, 69),
    n_seeds_to_undersample: int = 10,
    n_seeds_to_test_classifiers: int = 10,
    n_jobs: int = -1,
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
    "Accuracy", "ROC AUC", "F1 score", and "Balanced accuracy".
    """

    # NOTE: we still set a single seed, from which we generate a bunch of other
    # random seeds to be fed to the algorithm
    set_numpy_seed(generator_seeds[0])
    random_states_classifiers = randint(
        0, int(2**32 - 1), n_seeds_to_test_classifiers
    )

    set_numpy_seed(generator_seeds[1])
    random_states_undersampling = randint(0, int(2**32 - 1), n_seeds_to_undersample)

    x = x.reshape((x.shape[0], -1))
    data = DataFrame(x, index=folds)
    data["label"] = y

    results = []
    all_results: list[list[DataFrame]] = []
    for random_state_undersampling in tqdm(
        random_states_undersampling,
        desc="Random states undersampling progress",
        colour="red",
        disable=True if len(random_states_undersampling) <= 2 else False,
    ):
        data_resampled = data.groupby(axis=0, level=0).apply(
            resampling,
            resampling_method=kwargs.get("resampling_method", None),
            random_state=random_state_undersampling,
        )
        data_resampled.index = data_resampled.index.droplevel(1)

        x_resampled = data_resampled.drop(columns=["label"], inplace=False).values
        y_resampled = data_resampled["label"].values
        folds_resampled = data_resampled.index.get_level_values(0).values

        for random_state_classifier in tqdm(
            random_states_classifiers,
            desc="Random states classifiers progress:",
            colour="green",
        ):
            # TODO: we should iterate over different random states for the fold
            # generation as well, but independent from the random seeds for the algorithm
            classifier = LazyClassifier
            all_models = LOSO(
                groups_train=folds_resampled,
                groups_test=folds_resampled,
                x_train=x_resampled,
                x_test=x_resampled,
                y_train=y_resampled,
                y_test=y_resampled,
                ml_model=classifier,
                classifier_seed=random_state_classifier,
                n_jobs=n_jobs,
                classifiers=kwargs.get("classifiers", "all"),
                augment_data=kwargs.get("augment_data", False),
                augmentation_params=kwargs.get("augmentation_params", None),
                custom_fold_run_method=kwargs.get("custom_fold_run_method", None),
            )
            all_results.append(all_models)

            averages = (
                concat(all_models)
                .groupby(level=0)
                .mean()
                .sort_values(by="Accuracy", ascending=False)
            )
            standard_deviations = (
                concat(all_models)
                .groupby(level=0)
                .std()
                .sort_values(by="Accuracy", ascending=False)
            )
            standard_errors = standard_deviations / (len(set(folds.tolist())) ** 0.5)
            results.append(
                concat(
                    [averages, standard_errors],
                    axis=1,
                    keys=["Average", "Standard error"],
                )
            )

    averages_seeds = (
        concat(results)
        .groupby(level=0)
        .apply(lambda x: x.loc[:, IndexSlice["Average", :]].mean())
        .droplevel(axis=1, level=0)
        .sort_values(by=("Accuracy"), ascending=False)
    )

    errors_seeds = (
        concat(results)
        .groupby(level=0)
        .apply(
            lambda x: (x.loc[:, IndexSlice["Standard error", :]] ** 2).sum() ** 0.5
            / (n_seeds_to_test_classifiers * n_seeds_to_undersample)
        )
        .droplevel(axis=1, level=0)
        .sort_values(by="Accuracy", ascending=False)
    )
    return (
        concat(
            [averages_seeds, errors_seeds], axis=1, keys=["Average", "Standard error"]
        ),
        all_results,
    )


def under_sampling(
    x: ndarray,
    y: ndarray,
    folds: ndarray,
    resampling_method: Callable,
    random_state: int = 42,
) -> tuple[ndarray, ndarray, ndarray]:
    """
    Perform under-sampling using the given resampling method for each fold in the data.

    Parameters
    ----------
    x : ndarray
        The features of the data.
    y : ndarray
        The target labels of the data.
    folds : ndarray
        The array indicating which fold each sample belongs to.
    resampling_method : Callable
        The callable object used to resample the data for each fold.
    random_state : int, optional
        The seed used for the resampling, by default 42.

    Returns
    -------
    tuple[ndarray, ndarray, ndarray]
        A tuple containing the resampled features, target labels, and fold labels.
    """
    x = x.reshape((x.shape[0], -1))
    data = DataFrame(x, index=folds)
    data["label"] = y
    data_resampled = data.groupby(axis=0, level=0).apply(
        resampling, resampling_method=resampling_method, random_state=random_state
    )
    x_resampled = data_resampled.drop(columns=["label"], inplace=False).values
    y_resampled = data_resampled["label"].values
    folds_resampled = data_resampled.index.get_level_values(0).values

    return x_resampled, y_resampled, folds_resampled


def run_different_classifications(
    x_train: ndarray,
    x_test: ndarray,
    y_train: ndarray,
    y_test: ndarray,
    folds_train: int,
    folds_test: int,
    n_jobs: int = -1,
    n_seeds_to_test_classifiers: int = 30,
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
    x_train_resampled, y_train_resampled, folds_train_resampled = under_sampling(
        x_train, y_train, folds_train, RandomUnderSampler, random_state=42
    )
    x_test_resampled, y_test_resampled, folds_test_resampled = under_sampling(
        x_test, y_test, folds_test, RandomUnderSampler, random_state=42
    )

    results = []
    # NOTE: we still set a single seed, from which we generate a bunch of other
    # random seeds to be fed to the algorithm
    set_numpy_seed(42)

    random_states_classifiers = randint(
        0, int(2**32 - 1), n_seeds_to_test_classifiers
    )

    all_results: list[list[DataFrame]] = []
    for random_state_classifier in tqdm(
        random_states_classifiers,
        desc="Random states classifiers progress",
        colour="green",
    ):
        # TODO: we should iterate over different random states for the fold
        # generation as well, but independent from the random seeds for the algorithm
        classifier = LazyClassifier
        all_models = LOSO(
            groups_train=folds_train_resampled,
            groups_test=folds_test_resampled,
            x_train=x_train_resampled,
            x_test=x_test_resampled,
            y_train=y_train_resampled,
            y_test=y_test_resampled,
            ml_model=classifier,
            classifier_seed=random_state_classifier,
            n_jobs=n_jobs,
            classifiers=kwargs.get("classifiers", "all"),
        )
        all_results.append(all_models)

        averages = (
            pd.concat(all_models)
            .groupby(level=0)
            .mean()
            .sort_values(by="Accuracy", ascending=False)
        )
        standard_deviations = (
            pd.concat(all_models)
            .groupby(level=0)
            .std()
            .sort_values(by="Accuracy", ascending=False)
        )
        standard_errors = standard_deviations / 5**0.5
        results.append(
            pd.concat(
                [averages, standard_errors], axis=1, keys=["Average", "Standard error"]
            )
        )

    averages_seeds = (
        pd.concat(results)
        .groupby(level=0)
        .apply(lambda x: x.loc[:, IndexSlice["Average", :]].mean())
        .droplevel(axis=1, level=0)
        .sort_values(by=("Accuracy"), ascending=False)
    )

    errors_seeds = (
        pd.concat(results)
        .groupby(level=0)
        .apply(
            lambda x: (x.loc[:, IndexSlice["Standard error", :]] ** 2).sum() ** 0.5
            / (n_seeds_to_test_classifiers)
        )
        .droplevel(axis=1, level=0)
        .sort_values(by="Accuracy", ascending=False)
    )
    return (
        pd.concat(
            [averages_seeds, errors_seeds], axis=1, keys=["Average", "Standard error"]
        ),
        all_results,
    )


def run_opposite_side_prediction(
    features_right: ndarray,
    labels_right: ndarray,
    groups_right: ndarray,
    features_left: ndarray,
    labels_left: ndarray,
    groups_left: ndarray,
    which_comparison: str,
    generator_seeds: tuple[int, int] = [42, 666],
    n_seeds_to_test_classifiers: int = 10,
    n_seeds_to_undersample: int = 10,
    n_jobs: int = -1,
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
    set_numpy_seed(generator_seeds[0])
    random_states_classifiers = randint(
        0, int(2**32 - 1), n_seeds_to_test_classifiers
    )

    # NOTE: to avoid dependencies between the seeds for the classifiers and those
    # for the cross validation, two "main" seeds are required, from which then
    # generate all of the others. This also allows reproducibility of the code.
    set_numpy_seed(generator_seeds[1])
    random_states_undersample = randint(0, int(2**32 - 1), n_seeds_to_undersample)

    results = []
    all_results: list[list[DataFrame]] = []

    for random_state_undersample in tqdm(
        random_states_undersample,
        desc="Random states undersample progress:",
        colour="blue",
        disable=True if len(random_states_undersample) <= 2 else False,
    ):
        if which_comparison == "rxlx":
            (
                x_train_resampled,
                y_train_resampled,
                folds_train_resampled,
            ) = under_sampling(
                features_right,
                labels_right,
                groups_right,
                RandomUnderSampler,
                random_state=random_state_undersample,
            )
            x_test_resampled, y_test_resampled, folds_test_resampled = under_sampling(
                features_left,
                labels_left,
                groups_left,
                RandomUnderSampler,
                random_state=random_state_undersample,
            )
        elif which_comparison == "lxrx":
            (
                x_train_resampled,
                y_train_resampled,
                folds_train_resampled,
            ) = under_sampling(
                features_left,
                labels_left,
                groups_left,
                RandomUnderSampler,
                random_state=random_state_undersample,
            )
            x_test_resampled, y_test_resampled, folds_test_resampled = under_sampling(
                features_right,
                labels_right,
                groups_right,
                RandomUnderSampler,
                random_state=random_state_undersample,
            )
        else:
            raise ValueError(
                f"which_comparison must be either 'rxlx' or 'lxrx'. Received {which_comparison}"
            )

        for random_state_classifier in tqdm(
            random_states_classifiers,
            desc="Random states classifiers progress",
            colour="green",
        ):
            # TODO: we should iterate over different random states for the fold
            # generation as well, but independent from the random seeds for the algorithm
            classifier = LazyClassifier
            all_models = LOSO(
                groups_train=folds_train_resampled,
                groups_test=folds_test_resampled,
                x_train=x_train_resampled,
                x_test=x_test_resampled,
                y_train=y_train_resampled,
                y_test=y_test_resampled,
                ml_model=classifier,
                classifier_seed=random_state_classifier,
                n_jobs=n_jobs,
                classifiers=kwargs.get("classifiers", "all"),
            )
            all_results.append(all_models)

            averages = (
                pd.concat(all_models)
                .groupby(level=0)
                .mean()
                .sort_values(by="Accuracy", ascending=False)
            )
            standard_deviations = (
                pd.concat(all_models)
                .groupby(level=0)
                .std()
                .sort_values(by="Accuracy", ascending=False)
            )
            standard_errors = standard_deviations / (
                len(set(groups_right.tolist())) ** 0.5
            )
            results.append(
                pd.concat(
                    [averages, standard_errors],
                    axis=1,
                    keys=["Average", "Standard error"],
                )
            )

    averages_seeds = (
        pd.concat(results)
        .groupby(level=0)
        .apply(lambda x: x.loc[:, IndexSlice["Average", :]].mean())
        .droplevel(axis=1, level=0)
        .sort_values(by=("Accuracy"), ascending=False)
    )

    errors_seeds = (
        pd.concat(results)
        .groupby(level=0)
        .apply(
            lambda x: (x.loc[:, IndexSlice["Standard error", :]] ** 2).sum() ** 0.5
            / (n_seeds_to_test_classifiers * n_seeds_to_undersample)
        )
        .droplevel(axis=1, level=0)
        .sort_values(by="Accuracy", ascending=False)
    )
    return (
        pd.concat(
            [averages_seeds, errors_seeds], axis=1, keys=["Average", "Standard error"]
        ),
        all_results,
    )
