from collections import defaultdict
from logging import DEBUG, INFO, basicConfig, getLogger
from sys import path
from typing import Any
from json import dump

from pandas import HDFStore
from numpy import load

path.append(".")
from src.utils.io import load_config
from src.ml.cv import run_cross_validation_prediction, run_opposite_side_prediction

basicConfig(filename="logs/run_n_fold.log", level=DEBUG)

logger = getLogger("main")


def main():
    path_to_config: str = "src/run/ml/config_n_fold.yml"

    logger.info("Starting model training")
    configs: dict[str, Any] = load_config(path=path_to_config)
    logger.debug("Configs loaded")

    path_to_features_data: str = configs["path_to_features_data"]
    path_to_save_data_avgs: str = configs["path_to_save_data_avgs"]
    path_to_save_data_all: str = configs["path_to_save_data_all"]
    generator_seeds: tuple[int, int, int] = tuple(configs["generator_seeds"])
    n_seeds_to_test_classifiers: int = configs["n_seeds_to_test_classifiers"]
    n_seeds_to_test_folds: int = configs["n_seeds_to_test_folds"]
    n_seeds_to_undersample: int = configs["n_seeds_to_undersample"]
    n_fols: int = configs["n_fols"]
    n_jobs: int = configs["n_jobs"]

    data: dict[str, Any] = load(path_to_features_data)

    features_left = data["features_left"]
    features_right = data["features_right"]
    labels_left = data["labels_left"]
    labels_right = data["labels_right"]
    groups_left = data["groups_left"]
    groups_right = data["groups_right"]

    # if artifacts:
    #     artefacts_left = data["artefacts_left"]
    #     artefacts_right = data["artefacts_right"]

    averaged_results_cv, all_results_cv = dict(), dict()
    for side, side_features, side_labels, side_groups in zip(
        ["right", "left"],
        [features_right, features_left],
        [labels_right, labels_left],
        [groups_right, groups_left],
    ):
        (
            averaged_results_cv[side],
            all_results_cv[side],
        ) = run_cross_validation_prediction(
            x=side_features,
            y=side_labels,
            groups=side_groups,
            generator_seeds=generator_seeds,
            n_seeds_to_test_classifiers=n_seeds_to_test_classifiers,
            n_seeds_to_test_folds=n_seeds_to_test_folds,
            n_seeds_to_undersample=n_seeds_to_undersample,
            n_fols=n_fols,
            n_jobs=n_jobs,
        )

    for opposite_side in ["rxlx", "lxrx"]:
        (
            averaged_results_cv[opposite_side],
            all_results_cv[opposite_side],
        ) = run_opposite_side_prediction(
            features_right=features_right,
            labels_right=labels_right,
            groups_right=groups_right,
            features_left=features_left,
            labels_left=labels_left,
            groups_left=groups_left,
            which_comparison=opposite_side,
            generator_seeds=generator_seeds,
            n_seeds_to_test_classifiers=n_seeds_to_test_classifiers,
            n_seeds_to_undersample=n_seeds_to_undersample,
        )

    # Create an HDF5 file
    with HDFStore(path_to_save_data_avgs) as store:
        # Save each DataFrame in the dictionary to the HDF5 file
        for key, value in averaged_results_cv.items():
            store.put(key, value)
            
    # Create an HDF5 file
    with HDFStore(path_to_save_data_all) as store:
        # Save each DataFrame in the dictionary to the HDF5 file
        for key, value in all_results_cv.items():
            for i, el in enumerate(value):
                for j, el2 in enumerate(el):
                    store.put(f"{key}_{i}_{j}", el2)


if __name__ == "__main__":
    main()
