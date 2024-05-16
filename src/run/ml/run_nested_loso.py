from itertools import combinations, product
from json import dump
from logging import DEBUG, INFO, basicConfig, getLogger
from sys import path
from typing import Any

from numpy import load, ndarray, concatenate
from pandas import HDFStore
from tqdm.auto import tqdm

path.append(".")
import argparse

from src.ml.nested_loso import (
    run_opposite_side_prediction_updated,
    run_same_side_classifications,
)
from src.utils.io import load_config

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config_path",
    help="path for the config file used",
    default="src/run/ml/config_nested_loso.yml",
)
args = parser.parse_args()

basicConfig(filename="logs/run_nested_loso.log", level=DEBUG)

logger = getLogger("main")


def main():
    # path_to_config: str = "src/run/ml/config_nested_loso.yml"
    path_to_config: str = str(args.config_path)

    logger.info("Starting model training")
    configs: dict[str, Any] = load_config(path=path_to_config)
    logger.debug("Configs loaded")

    path_to_features_data: str = configs["path_to_features_data"]
    path_to_save_data_avgs: str = configs["path_to_save_data_avgs"]
    path_to_save_data_all: str = configs["path_to_save_data_all"]
    generator_seeds: tuple[int, int] = tuple(configs["generator_seeds"])
    n_seeds_to_test_classifiers: int = configs["n_seeds_to_test_classifiers"]
    n_seeds_to_undersample: int = configs["n_seeds_to_undersample"]
    n_seeds_to_test_folds: int = configs["n_seeds_to_test_folds"]
    n_folds_inner: int = configs["n_folds_inner"]
    n_jobs: int = configs["n_jobs"]
    debug_mode: bool = configs["debug_mode"]
    bilateral_fusion: bool = configs["bilateral_fusion"]

    subset_of_features: int = configs["subset_of_features"]
    path_to_feature_importance_list_left: str = configs[
        "path_to_feature_importance_list_left"
    ]
    path_to_feature_importance_list_right: str = configs[
        "path_to_feature_importance_list_right"
    ]

    print(f"Nested CV for dataset {path_to_features_data.split('/')[2]}")

    data: dict[str, Any] = load(path_to_features_data)

    features_left = data["features_left"]
    features_right = data["features_right"]
    labels_left = data["labels_left"]
    labels_right = data["labels_right"]
    groups_left = data["groups_left"]
    groups_right = data["groups_right"]

    features_left = features_left.reshape(features_left.shape[0], -1)
    features_right = features_right.reshape(features_right.shape[0], -1)

    if subset_of_features < 100:
        raise NotImplementedError(
            f"Only 100% of features are supported. {subset_of_features=}.\nIt was implemented in the previous version, but was removed to make space for bilateral fusion."
        )
        # important_features_left: DataFrame = read_csv(
        #     path_to_feature_importance_list_left
        # )
        # subset_of_features_num = int(
        #     subset_of_features * 0.01 * (len(important_features_left))
        # )
        # important_features_left: ndarray = important_features_left.iloc[
        #     :subset_of_features_num, 0
        # ].values

        # important_features_right: DataFrame = read_csv(
        #     path_to_feature_importance_list_right
        # )
        # subset_of_features_num = int(
        #     subset_of_features * 0.01 * (len(important_features_right))
        # )
        # important_features_right: ndarray = important_features_right.iloc[
        #     :subset_of_features_num, 0
        # ].values

        # features_left_opposite = features_left[:, important_features_right]
        # features_right_opposite = features_right[:, important_features_left]

        # features_left = features_left[:, important_features_left]

        # features_right = features_right[:, important_features_right]

    if not bilateral_fusion:
        complete_data = {
            "left": {
                "features": features_left,
                "labels": labels_left,
                "groups": groups_left,
            },
            "right": {
                "features": features_right,
                "labels": labels_right,
                "groups": groups_right,
            },
        }
    else:
        features_diff: ndarray = data["features_diff"]
        features_diff = features_diff.reshape(features_diff.shape[0], -1)

        labels_diff: ndarray = data["labels_diff"]
        groups_diff: ndarray = data["groups_diff"]

        features_left_right: ndarray = concatenate(
            (features_left, features_right), axis=1
        )
        features_left_diff: ndarray = concatenate(
            (features_left, features_diff), axis=1
        )
        features_right_diff: ndarray = concatenate(
            (features_right, features_diff), axis=1
        )
        features_all: ndarray = concatenate(
            (features_left, features_right, features_diff), axis=1
        )

        if (
            (labels_right == labels_left).all()
            and (labels_right == labels_diff).all()
            and (labels_left == labels_diff).all()
        ):
            logger.info("Labels are the same. Just using one for combination")
            labels_left_right: ndarray = labels_left
            labels_left_diff: ndarray = labels_left
            labels_right_diff: ndarray = labels_left
            labels_all: ndarray = labels_left
        else:
            raise RuntimeError(
                f"Labels are not the same. Cannot combine them.\nThese are the labels: {labels_left=}\n {labels_right=}\n {labels_diff=}"
            )

        if (
            (groups_right == groups_left).all()
            and (groups_right == groups_diff).all()
            and (groups_left == groups_diff).all()
        ):
            logger.info("Labels are the same. Just using one for combination")
            groups_left_right: ndarray = groups_left
            groups_left_diff: ndarray = groups_left
            groups_right_diff: ndarray = groups_left
            groups_all: ndarray = groups_left
        else:
            raise RuntimeError(
                f"Groups are not the same. Cannot combine them.\nThese are the groups: {groups_left=}\n {groups_right=}\n {groups_diff=}"
            )
        complete_data = {
            "left": {
                "features": features_left if not debug_mode else features_left[:200],
                "labels": labels_left if not debug_mode else labels_left[:200],
                "groups": groups_left if not debug_mode else groups_left[:200],
            },
            "right": {
                "features": features_right if not debug_mode else features_right[:200],
                "labels": labels_right if not debug_mode else labels_right[:200],
                "groups": groups_right if not debug_mode else groups_right[:200],
            },
            "diff": {
                "features": features_diff if not debug_mode else features_diff[:200],
                "labels": labels_diff if not debug_mode else labels_diff[:200],
                "groups": groups_diff if not debug_mode else groups_diff[:200],
            },
            "left+right": {
                "features": features_left_right
                if not debug_mode
                else features_left_right[:200],
                "labels": labels_left_right
                if not debug_mode
                else labels_left_right[:200],
                "groups": groups_left_right
                if not debug_mode
                else groups_left_right[:200],
            },
            "left+diff": {
                "features": features_left_diff
                if not debug_mode
                else features_left_diff[:200],
                "labels": labels_left_diff
                if not debug_mode
                else labels_left_diff[:200],
                "groups": groups_left_diff
                if not debug_mode
                else groups_left_diff[:200],
            },
            "right+diff": {
                "features": features_right_diff
                if not debug_mode
                else features_right_diff[:200],
                "labels": labels_right_diff
                if not debug_mode
                else labels_right_diff[:200],
                "groups": groups_right_diff
                if not debug_mode
                else groups_right_diff[:200],
            },
            "left+right+diff": {
                "features": features_all if not debug_mode else features_all[:200],
                "labels": labels_all if not debug_mode else labels_all[:200],
                "groups": groups_all if not debug_mode else groups_all[:200],
            },
        }

    averaged_results_cv, all_results_cv = dict(), dict()
    for side, side_data in (
        pbar := tqdm(
            complete_data.items(),
            position=2,
            leave=True,
        )
    ):
        pbar.set_description(f"Same side progress (current {side})")
        (
            averaged_results_cv[side],
            all_results_cv[side],
        ) = run_same_side_classifications(
            x=side_data["features"],
            y=side_data["labels"],
            folds=side_data["groups"],
            generator_seeds=generator_seeds,
            n_seeds_to_test_folds=n_seeds_to_test_folds,
            n_seeds_to_test_classifiers=n_seeds_to_test_classifiers,
            n_seeds_to_undersample=n_seeds_to_undersample,
            n_inner_folds=n_folds_inner,
            n_jobs=n_jobs,
        )

    opposite_sides: list[str] = list(combinations(list(complete_data.keys()), r=2))
    # NOTE: we removed when combinations are on the test side, while the train is given by a single
    # value, in order to avoid problems with the fold selection.
    bad_sides = list(
        product(
            ["left", "right", "diff"],
            ["left+right", "left+diff", "right+diff", "left+right+diff"],
        )
    )
    bad_sides.extend(
        list(
            product(
                ["left+right", "left+diff", "right+diff", "left+right+diff"],
                ["left", "right", "diff"],
            )
        )
    )
    bad_sides.extend(
        list(
            product(
                [el for el in complete_data.keys() if el != "left+right+diff"],
                ["left+right+diff"],
            )
        )
    )
    bad_sides.extend(
        list(
            product(
                ["left+right+diff"],
                [el for el in complete_data.keys() if el != "left+right+diff"],
            )
        )
    )
    opposite_sides = [side for side in opposite_sides if side not in bad_sides]

    for opposite_side in (
        pbar := tqdm(
            opposite_sides,
            position=2,
            leave=True,
        )
    ):
        (
            averaged_results_cv[str(opposite_side)],
            all_results_cv[str(opposite_side)],
        ) = run_opposite_side_prediction_updated(
            features_train=complete_data[opposite_side[0]]["features"],
            labels_train=complete_data[opposite_side[0]]["labels"],
            groups_train=complete_data[opposite_side[0]]["groups"],
            features_test=complete_data[opposite_side[1]]["features"],
            labels_test=complete_data[opposite_side[1]]["labels"],
            groups_test=complete_data[opposite_side[1]]["groups"],
            generator_seeds=generator_seeds,
            n_seeds_to_test_classifiers=n_seeds_to_test_classifiers,
            n_seeds_to_undersample=n_seeds_to_undersample,
            n_seeds_to_test_folds=n_seeds_to_test_folds,
            n_jobs=n_jobs,
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
