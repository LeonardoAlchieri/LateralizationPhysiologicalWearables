from collections import defaultdict
from logging import DEBUG, INFO, basicConfig, getLogger
from sys import path
from typing import Any
from json import dump

from effect_size_analysis.cliff_delta import cliff_delta
from numpy import load
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from joblib_progress import joblib_progress

path.append(".")
from src.feature_extraction.eda import EDA_FEATURE_NAMES
from src.utils.io import load_config

basicConfig(filename="logs/run_cliff_delta_features.log", level=DEBUG)

logger = getLogger("main")

CLIFF_DELTA_BINS = {
    "small": 0.11,
    "medium": 0.28,
    "large": 0.43,
}  # effect sizes from (Vargha and Delaney (2000)) "negligible" for the rest


def compute_cliff_delta(
    class_name, data_left, data_right, num_component, component_name, feature, alpha, i
):
    result = cliff_delta(
        s1=data_left[:, i, num_component],
        s2=data_right[:, i, num_component],
        alpha=alpha,
        accurate_ci=True,
        raise_nan=False,
    )
    return (class_name, component_name, feature, result)


def main():
    path_to_config: str = "src/run/statistical_analysis/config_cliff_delta_features.yml"

    logger.info("Starting model training")
    configs: dict[str, Any] = load_config(path=path_to_config)
    logger.debug("Configs loaded")

    path_to_features_data: str = configs["path_to_features_data"]
    path_to_save_data: str = configs["path_to_save_data"]
    # artifacts: bool = configs["artifacts"]
    alpha: int = configs.get("alpha", 0.05)
    components: list[str] = configs["components"]
    n_jobs: int = configs["n_jobs"]

    data: dict[str, Any] = load(path_to_features_data)

    features_left = data["features_left"]
    features_right = data["features_right"]
    labels_left = data["labels_left"]
    labels_right = data["labels_right"]
    # groups_left = data["groups_left"]
    # groups_right = data["groups_right"]

    # if artifacts:
    #     artefacts_left = data["artefacts_left"]
    #     artefacts_right = data["artefacts_right"]

    positive_data_left = features_left[labels_left == 1]
    negative_data_left = features_left[labels_left == 0]

    positive_data_right = features_right[labels_right == 1]
    negative_data_right = features_right[labels_right == 0]

    cliff_delta_results = defaultdict(lambda: defaultdict(lambda: dict()))
    # Parallelize the computation
    with joblib_progress(
        "Random seed iterations", total=len(EDA_FEATURE_NAMES) * len(components) * 2
    ):
        results = Parallel(n_jobs=n_jobs)(
            delayed(compute_cliff_delta)(
                class_name,
                data_left,
                data_right,
                num_component,
                component_name,
                feature,
                alpha,
                i,
            )
            for class_name, data_left, data_right in [
                ("positive", positive_data_left, positive_data_right),
                ("negative", negative_data_left, negative_data_right),
            ]
            for num_component, component_name in enumerate(components)
            for i, feature in enumerate(EDA_FEATURE_NAMES)
        )
        # Store the results in the cliff_delta_results dictionary
        for class_name, component_name, feature, result in results:
            cliff_delta_results[class_name][component_name][feature] = result

    # cliff_delta_results = defaultdict(lambda: defaultdict(lambda: dict()))
    # for class_name, data_left, data_right in [
    #     ("positive", positive_data_left, positive_data_right),
    #     ("negative", negative_data_left, negative_data_right),
    # ]:
    #     for num_component, component_name in enumerate(components):
    #         for i, feature in tqdm(
    #             enumerate(EDA_FEATURE_NAMES),
    #             desc=f"Feature progress for component {component_name}, class {class_name}",
    #             total=len(EDA_FEATURE_NAMES),
    #         ):
    #             cliff_delta_results[class_name][component_name][feature] = cliff_delta(
    #                 s1=data_left[:, i, num_component],
    #                 s2=data_right[:, i, num_component],
    #                 alpha=alpha,
    #                 accurate_ci=True,
    #                 raise_nan=False,
    #             )

    # save dictionary to json file
    with open(path_to_save_data, "w") as f:
        dump(cliff_delta_results, f)
    # d = {"positive": cliff_delta_positive, "negative": cliff_delta_negative}
    # cliff_deltas = concat_dataframe(d.values(), axis=0, keys=d.keys())

    # cliff_deltas.to_csv(path_to_save_data)


if __name__ == "__main__":
    main()
