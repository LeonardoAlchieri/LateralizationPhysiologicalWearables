from itertools import product

# setup logger
from logging import getLogger
from signal import SIGALRM
from signal import alarm as timeout_alarm
from signal import signal
from typing import Any, Callable, Generator, Iterable

import pandas as pd
from joblib import Parallel, delayed, parallel_backend
from joblib_progress import joblib_progress
from lazypredict.Supervised import (
    categorical_transformer_high,
    categorical_transformer_low,
    get_card_split,
    numeric_transformer,
)
from numpy import nan, ndarray
from numpy import number as npnumber
from numpy.random import randint
from numpy.random import seed as set_numpy_seed
from pandas import DataFrame, IndexSlice
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_halving_search_cv
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import HalvingRandomSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from tqdm.auto import tqdm

from ray.util.joblib import register_ray

from src.ml import local_resampling, resampling
from src.ml.classifier_list import CLASSIFIERS_HYPERPARAMETER_LIST

logger = getLogger("nested")

import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_DYNAMIC"] = "FALSE"


def timeout_handler(signum, frame):
    print("Classifier got stuck. Raising exception.")
    raise TimeoutError("Stuck classifier")


def preprocessing(
    X_train: ndarray | DataFrame,
):
    if isinstance(X_train, ndarray):
        X_train = pd.DataFrame(X_train)

    # stole this from LazyPredict library
    numeric_features = X_train.select_dtypes(include=[npnumber]).columns
    categorical_features = X_train.select_dtypes(include=["object"]).columns

    categorical_low, categorical_high = get_card_split(X_train, categorical_features)

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical_low", categorical_transformer_low, categorical_low),
            ("categorical_high", categorical_transformer_high, categorical_high),
        ]
    )
    return preprocessor


def perform_grid_search_estimation(
    classifier: ClassifierMixin,
    random_state_classifier: int,
    preprocessor: Pipeline,
    search_space: dict[str, Any],
    folds_inner: list,
    x_train: ndarray,
    y_train: ndarray,
    x_test: ndarray,
    y_test: ndarray,
    max_resources: int | str = "auto",
):
    if "random_state" in classifier().get_params().keys():
        model: ClassifierMixin = classifier(random_state=random_state_classifier)
    else:
        model: ClassifierMixin = classifier()

    # execute search
    clf = make_pipeline(
        preprocessor,
        HalvingRandomSearchCV(
            model,
            search_space,
            scoring="accuracy",
            cv=folds_inner,
            refit=True,
            max_resources=max_resources,
            n_jobs=1,
            random_state=random_state_classifier,
            # error_score="raise",
        ),
    )
    result = clf.fit(x_train.copy(), y_train.copy())
    yhat = result.predict(x_test)
    acc = balanced_accuracy_score(y_test, yhat)
    return classifier.__name__, acc
    # models[classifier.__name__] = acc


def single_classifier_training(
    classifier: ClassifierMixin,
    random_state_classifier: int,
    preprocessor: Pipeline,
    search_space: dict[str, Any],
    folds_inner: list,
    x_train: ndarray,
    y_train: ndarray,
    x_test: ndarray,
    y_test: ndarray,
    max_resources: int | str = "auto",
    timeout: int | None = None,
):
    if timeout is not None:
        signal(SIGALRM, timeout_handler)
        timeout_alarm(timeout)

        try:
            result = perform_grid_search_estimation(
                classifier=classifier,
                random_state_classifier=random_state_classifier,
                preprocessor=preprocessor,
                search_space=search_space,
                folds_inner=folds_inner,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                max_resources=max_resources,
            )
        except TimeoutError as exc:
            print(exc)
            print(f"Classifier stuck: {classifier.__name__}")
            result = classifier.__name__, nan
    else:
        result = perform_grid_search_estimation(
            classifier=classifier,
            random_state_classifier=random_state_classifier,
            preprocessor=preprocessor,
            search_space=search_space,
            folds_inner=folds_inner,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            max_resources=max_resources,
        )
    return result


def fit_with_hyperparameters(
    x_train: ndarray,
    y_train: ndarray,
    x_test: ndarray,
    y_test: ndarray,
    random_state_classifier: int,
    random_state_fold: int,
    n_inner_folds: int,
    max_resources: int | str = "auto",
    timeout: int | None = None,
) -> DataFrame:
    folds_inner: Generator = StratifiedKFold(
        n_splits=n_inner_folds, random_state=random_state_fold, shuffle=True
    ).split(x_train, y_train)
    folds_inner = list(folds_inner)

    preprocessor = preprocessing(x_train)

    models: list[tuple[str, float]] = [
        single_classifier_training(
            classifier=classifier,
            random_state_classifier=random_state_classifier,
            preprocessor=preprocessor,
            search_space=search_space,
            folds_inner=folds_inner,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            max_resources=max_resources,
            timeout=timeout,
        )
        for classifier, search_space in CLASSIFIERS_HYPERPARAMETER_LIST.items()
    ]
    models: dict[str, float] = {k: v for k, v in models}

    models = DataFrame.from_dict(models, orient="index", columns=["Balanced Accuracy"])
    return models


def resample_train_data(
    x_train: ndarray,
    y_train: ndarray,
    groups_train: list,
    resampling_method: Callable,
    random_state_undersampling: int,
):
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
    return x_resampled_train, y_resampled_train


def run_hyper_fold(
    train_index: Iterable,
    test_index: Iterable,
    x_full: ndarray,
    y_full: ndarray,
    groups: ndarray,
    random_state_classifier: int,
    random_state_undersampling: int,
    random_state_fold: int,
    n_inner_folds: int = 3,
    resampling_method: Callable | None = None,
    **kwargs,
):
    x_test: ndarray
    x_train: ndarray
    y_test: ndarray
    y_train: ndarray
    x_train, x_test = x_full[train_index], x_full[test_index]
    y_train, y_test = y_full[train_index], y_full[test_index]
    groups_train, groups_test = groups[train_index], groups[test_index]

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
        max_resources=kwargs.get("max_resources", "auto"),
        timeout=kwargs.get("timeout", None),
    )

    return models


def compute_outer_folds_same_side(
    data: DataFrame,
    n_outer_folds: int,
    n_inner_folds: int,
    random_state_fold: int,
    random_state_classifier: int,
    random_state_undersampling: int,
    classifiers: list[ClassifierMixin],
    resampling_method: Callable | None,
    **kwargs,
):
    x_full: ndarray = data.drop(columns=["label"], inplace=False).values
    y_full: ndarray = data["label"].values
    groups: ndarray = data.index.get_level_values(0).values

    # NOTE: the fold generation should be fixed, to limit the accuracy
    # be due exclusively to starting confitions in the algorithm
    folds: Generator = StratifiedKFold(
        n_splits=n_outer_folds, random_state=random_state_fold, shuffle=True
    ).split(x_full, y_full)

    folds = list(folds)

    custom_method: Callable | None = kwargs.get("custom_fold_run_method", None)
    if custom_method is None:
        all_models: list[DataFrame] = [
            run_hyper_fold(
                train_index=train_index,
                test_index=test_index,
                x_full=x_full.copy(),
                y_full=y_full.copy(),
                groups=groups.copy(),
                random_state_classifier=random_state_classifier,
                random_state_undersampling=random_state_undersampling,
                random_state_fold=random_state_fold,
                n_inner_folds=n_inner_folds,
                classifiers=classifiers,
                resampling_method=resampling_method,
                max_resources=kwargs.get("max_resources", "auto"),
                timeout=kwargs.get("timeout", None),
            )
            for train_index, test_index in folds
        ]
    else:
        raise NotImplementedError("Custom nested fold run method not implemented yet.")

    averages = (
        pd.concat(all_models)
        .groupby(level=0)
        .mean()
        # .sort_index(ascending=False)
        .sort_values(by="Balanced Accuracy", ascending=False)
    )
    # standard_deviations = (
    #     pd.concat(all_models)
    #     .groupby(level=0)
    #     .std()
    #     .sort_values(by="Balanced Accuracy", ascending=False)
    # )
    standard_errors = (
        pd.concat(all_models)
        .groupby(level=0)
        .sem()
        # .sort_index(ascending=False)
        .sort_values(by="Balanced Accuracy", ascending=False)
    )
    # standard_errors = standard_deviations / (n_outer_folds**0.5)

    return all_models, pd.concat(
        [averages, standard_errors],
        axis=1,
        keys=["Average", "Standard error"],
    )


def run_nested_cross_validation_prediction(
    x: ndarray,
    y: ndarray,
    groups: ndarray,
    generator_seeds: tuple[int, int, int] = (42, 666, 69),
    n_seeds_to_test_classifiers: int = 10,
    n_seeds_to_test_folds: int = 10,
    n_seeds_to_undersample: int = 10,
    n_inner_folds: int = 3,
    n_outer_folds: int = 5,
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

    results: list[DataFrame] = []
    all_results: list[list[DataFrame]] = []

    possible_combinations = list(
        product(
            random_states_folds, random_states_undersampling, random_states_classifiers
        )
    )

    register_ray()
    
    with joblib_progress(
        "Random seed iterations", total=len(list(possible_combinations))
    ):
        with parallel_backend("ray"):
            outer_folds_output: list[tuple[list, list[list[DataFrame]]]] = Parallel(
                n_jobs=kwargs.get("n_jobs", 1),
            )(
                delayed(compute_outer_folds_same_side)(
                    data=data,
                    n_outer_folds=n_outer_folds,
                    n_inner_folds=n_inner_folds,
                    random_state_fold=random_state_fold,
                    random_state_classifier=random_state_classifier,
                    random_state_undersampling=random_state_undersampling,
                    classifiers=kwargs.get("classifiers", "all"),
                    resampling_method=kwargs.get("resampling_method", None),
                    max_resources=kwargs.get("max_resources", "auto"),
                    timeout=kwargs.get("timeout", None),
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
            lambda x: ((x.loc[:, IndexSlice["Standard error", :]] ** 2).sum()) ** 0.5
            / (
                n_seeds_to_test_classifiers
                * n_seeds_to_test_folds
                * n_seeds_to_undersample
            )
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


def compute_outer_folds_opposite_side(
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
    **kwargs,
) -> tuple[list[DataFrame], list[list[DataFrame]]]:
    x_resampled_rx, y_resampled_rx, _ = local_resampling(
        features_right,
        labels_right,
        groups_right,
        seed=random_state_undersampling,
    )
    x_resampled_lx, y_resampled_lx, _ = local_resampling(
        features_left, labels_left, groups_left, seed=random_state_undersampling
    )
    if which_comparison == "rxlx":
        models = fit_with_hyperparameters(
            x_train=x_resampled_rx,
            y_train=y_resampled_rx,
            x_test=features_left.reshape((features_left.shape[0], -1)),
            y_test=labels_left.reshape((labels_left.shape[0], -1)),
            random_state_classifier=random_state_classifier,
            random_state_fold=random_state_fold,
            n_inner_folds=n_inner_folds,
            max_resources=kwargs.get("max_resources", "auto"),
            timeout=kwargs.get("timeout", None),
        )
    elif which_comparison == "lxrx":
        models = fit_with_hyperparameters(
            x_train=x_resampled_lx,
            y_train=y_resampled_lx,
            x_test=features_right.reshape((features_right.shape[0], -1)),
            y_test=labels_right.reshape((labels_right.shape[0], -1)),
            random_state_classifier=random_state_classifier,
            random_state_fold=random_state_fold,
            n_inner_folds=n_inner_folds,
            max_resources=kwargs.get("max_resources", "auto"),
            timeout=kwargs.get("timeout", None),
        )
    else:
        raise ValueError(
            f"which_comparison must be either 'rxlx' or 'lxrx'. Received {which_comparison}"
        )

    return [models], pd.concat(
        [models],
        axis=1,
        keys=["Average"],
    )


def run_opposite_side_prediction_hyper(
    features_right: ndarray,
    labels_right: ndarray,
    groups_right: ndarray,
    features_left: ndarray,
    labels_left: ndarray,
    groups_left: ndarray,
    which_comparison: str,
    generator_seeds: tuple[int, int, int] = (42, 666, 69),
    n_seeds_to_test_folds: int = 10,
    n_seeds_to_test_classifiers: int = 10,
    n_seeds_to_undersample: int = 10,
    n_inner_folds: int = 3,
    **kwargs,
):
    set_numpy_seed(generator_seeds[0])
    random_states_classifiers = randint(
        0, int(2**32 - 1), n_seeds_to_test_classifiers
    )

    # NOTE: to avoid dependencies between the seeds for the classifiers and those
    # for the cross validation, two "main" seeds are required, from which then
    # generate all of the others. This also allows reproducibility of the code.
    set_numpy_seed(generator_seeds[1])
    random_states_undersampling = randint(0, int(2**32 - 1), n_seeds_to_undersample)

    set_numpy_seed(generator_seeds[2])
    random_states_folds = randint(0, int(2**32 - 1), n_seeds_to_test_folds)

    # results = []
    # all_results: list[list[DataFrame]] = []
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
            delayed(compute_outer_folds_opposite_side)(
                features_right=features_right,
                labels_right=labels_right,
                groups_right=groups_right,
                features_left=features_left,
                labels_left=labels_left,
                groups_left=groups_left,
                random_state_undersampling=random_state_undersampling,
                random_state_classifier=random_state_classifier,
                random_state_fold=random_state_fold,
                n_inner_folds=n_inner_folds,
                which_comparison=which_comparison,
                max_resources=kwargs.get("max_resources", "auto"),
                timeout=kwargs.get("timeout", None),
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
            lambda x: x.loc[:, IndexSlice["Average", :]].sem()
            # lambda x: x.loc[:, IndexSlice["Average", :]].std()
            # / (
            #     (
            #         n_seeds_to_test_classifiers
            #         * n_seeds_to_undersample
            #         * n_seeds_to_test_folds
            #     )
            #     ** 0.5
            # )
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
