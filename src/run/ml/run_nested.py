from typing import Any, Literal
from collections import defaultdict
from logging import DEBUG, INFO, basicConfig, getLogger
from sys import path
from typing import Any
from json import dump

from pandas import HDFStore, read_csv, DataFrame
from numpy import load, ndarray

path.append(".")
from src.utils.io import load_config
from src.ml.nested import run_nested_cross_validation_prediction, run_opposite_side_prediction_hyper

basicConfig(filename="logs/run_nested.log", level=DEBUG)

logger = getLogger("main")


def main():
    path_to_config: str = "src/run/ml/config_nested.yml"

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
    n_folds_outer: int = configs["n_folds_outer"]
    n_folds_inner: int = configs["n_folds_inner"]
    n_jobs: int = configs["n_jobs"]
    timeout: int = configs["timeout"]
    max_resources: int = configs["max_resources"]
    n_candidates: int | Literal['exhaust'] = configs["n_candidates"]
    debug_mode: bool = configs["debug_mode"]
    
    subset_of_features: int = configs["subset_of_features"]
    path_to_feature_importance_list_left: str = configs['path_to_feature_importance_list_left']
    path_to_feature_importance_list_right: str = configs['path_to_feature_importance_list_right']
    
    print(f"Nested CV for dataset {path_to_features_data.split('/')[2]}")

    data: dict[str, Any] = load(path_to_features_data)

    if debug_mode:
        features_left = data["features_left"][:200]
        features_right = data["features_right"][:200]
        labels_left = data["labels_left"][:200]
        labels_right = data["labels_right"][:200]
        groups_left = data["groups_left"][:200]
        groups_right = data["groups_right"][:200]
    else:
        features_left: ndarray = data["features_left"]
        features_right: ndarray = data["features_right"]
        labels_left: ndarray = data["labels_left"]
        labels_right: ndarray = data["labels_right"]
        groups_left: ndarray = data["groups_left"]
        groups_right: ndarray = data["groups_right"]

    features_left = features_left.reshape(features_left.shape[0], -1)
    features_right = features_right.reshape(features_right.shape[0], -1)
    if subset_of_features < 100:
        
        important_features_left: DataFrame = read_csv(path_to_feature_importance_list_left)
        subset_of_features_num = int(subset_of_features * 0.01 * (len(important_features_left)))
        important_features_left: ndarray = important_features_left.iloc[:subset_of_features_num, 0].values
        
        important_features_right: DataFrame = read_csv(path_to_feature_importance_list_right)
        subset_of_features_num = int(subset_of_features * 0.01 * (len(important_features_right)))
        important_features_right: ndarray = important_features_right.iloc[:subset_of_features_num, 0].values
        
        features_left = features_left[: ,important_features_left]
        
        features_right = features_right[: ,important_features_right]
        
    
    print("\n")
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
        print(f'Starting {side} side')
        (
            averaged_results_cv[side],
            all_results_cv[side],
        ) = run_nested_cross_validation_prediction(
            x=side_features,
            y=side_labels,
            groups=side_groups,
            generator_seeds=generator_seeds,
            n_seeds_to_test_classifiers=n_seeds_to_test_classifiers,
            n_seeds_to_test_folds=n_seeds_to_test_folds,
            n_seeds_to_undersample=n_seeds_to_undersample,
            n_inner_folds=n_folds_inner,
            n_outer_folds=n_folds_outer,
            n_jobs=n_jobs,
            timeout=timeout,
            max_resources=max_resources,
            n_candidate=n_candidates,
            important_features_left=important_features_left,
            important_features_right=important_features_right,
        )


    print("\n")
    for opposite_side in ["rxlx", "lxrx"]:
        print(f'Starting {opposite_side} opposite side')
        (
            averaged_results_cv[opposite_side],
            all_results_cv[opposite_side],
        ) = run_opposite_side_prediction_hyper(
            features_right=features_right,
            labels_right=labels_right,
            groups_right=groups_right,
            features_left=features_left,
            labels_left=labels_left,
            groups_left=groups_left,
            which_comparison=opposite_side,
            generator_seeds=generator_seeds,
            n_seeds_to_test_folds=n_seeds_to_test_folds,
            n_seeds_to_test_classifiers=n_seeds_to_test_classifiers,
            n_seeds_to_undersample=n_seeds_to_undersample,
            n_jobs=n_jobs,
            timeout=timeout,
            max_resources=max_resources,
            n_candidates=n_candidates,
            important_features_left=important_features_left,
            important_features_right=important_features_right,
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
