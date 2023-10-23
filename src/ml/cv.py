from typing import Callable, Iterable
from itertools import product

import pandas as pd
from joblib import Parallel, delayed
from lazypredict.Supervised import LazyClassifier
from numpy import ndarray
from numpy.random import randint
from numpy.random import seed as set_numpy_seed
from pandas import DataFrame, IndexSlice
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm

from src.ml import resampling, local_resampling


def run_fold(
    train_index: Iterable,
    test_index: Iterable,
    x_full: ndarray,
    y_full: ndarray,
    groups: ndarray,
    random_state_classifier: int,
    random_state_undersampling: int,
    resampling_method: Callable | None = None,
    **kwargs,
):
    x_train, x_test = x_full[train_index], x_full[test_index]
    y_train, y_test = y_full[train_index], y_full[test_index]
    groups_train, groups_test = groups[train_index], groups[test_index]

    x_train = x_train.reshape((x_train.shape[0], -1))
    data_train = DataFrame(x_train, index=groups_train)
    data_train["label"] = y_train

    data_resampled_train = data_train.groupby(axis=0, level=0).apply(
        resampling,
        resampling_method=resampling_method,
        random_state=random_state_undersampling,
    )
    data_resampled_train.index = data_resampled_train.index.droplevel(1)
    x_resampled_train: ndarray = data_resampled_train.drop(
        columns=["label"], inplace=False
    ).values
    y_resampled_train: ndarray = data_resampled_train["label"].values

    clf = LazyClassifier(
        predictions=True,
        random_state=random_state_classifier,
        classifiers=kwargs.get("classifiers", "all"),
    )
    models, _ = clf.fit(x_resampled_train, x_test, y_resampled_train, y_test)
    return models


def run_cross_validation_prediction(
    x: ndarray,
    y: ndarray,
    groups: ndarray,
    generator_seeds: tuple[int, int, int] = (42, 666, 69),
    n_seeds_to_test_classifiers: int = 10,
    n_seeds_to_test_folds: int = 10,
    n_seeds_to_undersample: int = 10,
    n_fols: int = 5,
    **kwargs,
) -> DataFrame:
    """
    Run N-fold cross validation for a classification task using LazyClassifier library.

    Parameters
    ----------
    x : ndarray
        Array of features to be used for prediction.
    y : ndarray
        Array of labels to be predicted.
    groups : ndarray
        Array of groups used for resampling the data.
    generator_seeds : list[int], optional
        List of generator seeds to use for generating random states for classifiers and folds, by default [42, 666].
    n_seeds_to_test_classifiers : int, optional
        Number of random seeds to use for testing classifiers, by default 10.
    n_seeds_to_test_folds : int, optional
        Number of random seeds to use for testing folds, by default 10.
    n_fols : int, optional
        Number of folds to use in cross validation, by default 5.
    **kwargs
        Additional arguments to be passed to the resampling function, such as resampling method and resampling random state.

    Returns
    -------
    results : DataFrame
        A dataframe with the average and standard error of the cross validation results, sorted by accuracy.
    all_results : list[list[DataFrame]]
        A list of dataframes containing the results for each seed combination and fold.

    Examples
    --------
    >>> results, all_results = run_cross_validation_prediction(x_train, y_train, groups_train)
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
    data = DataFrame(x, index=groups)
    data["label"] = y

    results = []
    all_results: list[list[DataFrame]] = []

    for random_state_folds, random_state_undersampling in tqdm(
        product(random_states_folds, random_states_undersampling),
        desc="Random states (and undersampling) folds progress",
        colour="blue",
        disable=True
        if len(product(random_states_folds, random_states_undersampling)) <= 2
        else False,
    ):
        x_full: ndarray = data.drop(columns=["label"], inplace=False).values
        y_full: ndarray = data["label"].values
        groups: ndarray = data.index.get_level_values(0).values

        # NOTE: the fold generation should be fixed, to limit the accuracy
        # be due exclusively to starting confitions in the algorithm
        folds = StratifiedKFold(
            n_splits=n_fols, random_state=random_state_folds, shuffle=True
        ).split(x_full, y_full)

        for random_state_classifier in tqdm(
            random_states_classifiers,
            desc="Random states classifiers progress",
            colour="green",
            disable=True if len(random_states_classifiers) <= 2 else False,
        ):
            custom_method: Callable | None = kwargs.get("custom_fold_run_method", None)
            if custom_method is None:
                all_models: list[DataFrame] = Parallel(n_jobs=kwargs.get("n_jobs", -1))(
                    delayed(run_fold)(
                        train_index,
                        test_index,
                        x_full,
                        y_full,
                        groups,
                        random_state_classifier,
                        classifiers=kwargs.get("classifiers", "all"),
                        resampling_method=kwargs.get("resampling_method", None),
                    )
                    for train_index, test_index in folds
                )
            else:
                custom_method: Callable
                all_models: list[DataFrame] = Parallel(n_jobs=kwargs.get("n_jobs", -1))(
                    delayed(custom_method)(
                        train_index,
                        test_index,
                        x_full,
                        y_full,
                        groups,
                        random_state_classifier,
                        classifiers=kwargs.get("classifiers", "all"),
                    )
                    for train_index, test_index in folds
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
            standard_errors = standard_deviations / (n_fols**0.5)
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
            lambda x: ((x.loc[:, IndexSlice["Standard error", :]] ** 2).sum()) ** 0.5
            / (
                n_seeds_to_test_classifiers
                * n_seeds_to_test_folds
                * n_seeds_to_undersample
            )
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
):
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
        for random_state_classifier in tqdm(
            random_states_classifiers,
            desc="Random states classifiers progress:",
            colour="green",
            disable=True if len(random_states_classifiers) <= 2 else False,
        ):
            clf = LazyClassifier(predictions=True, random_state=random_state_classifier)

            x_resampled_rx, y_resampled_rx, _ = local_resampling(
                features_right,
                labels_right,
                groups_right,
                seed=random_state_undersample,
            )
            x_resampled_lx, y_resampled_lx, _ = local_resampling(
                features_left, labels_left, groups_left, seed=random_state_undersample
            )
            if which_comparison == "rxlx":
                models, _ = clf.fit(
                    x_resampled_rx, features_left, y_resampled_rx, labels_left
                )
            elif which_comparison == "lxrx":
                models, _ = clf.fit(
                    x_resampled_lx, features_right, y_resampled_lx, labels_right
                )
            else:
                raise ValueError(
                    f"which_comparison must be either 'rxlx' or 'lxrx'. Received {which_comparison}"
                )

            all_results.append([models])
            results.append(
                pd.concat(
                    [models],
                    axis=1,
                    keys=["Average"],
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
            lambda x: x.loc[:, IndexSlice["Average", :]].std()
            / ((n_seeds_to_test_classifiers * n_seeds_to_undersample) ** 0.5)
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
