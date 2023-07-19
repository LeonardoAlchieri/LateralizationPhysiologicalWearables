from collections import defaultdict
from logging import DEBUG, INFO, basicConfig, getLogger
from sys import path
from typing import Any
from json import dump

from mpsci.stats import pearsonr_ci
from scipy.stats import pearsonr
from numpy import load, ndarray
from tqdm.auto import tqdm

path.append(".")
from src.feature_extraction.eda import EDA_FEATURE_NAMES
from src.utils.io import load_config

basicConfig(filename="logs/run_correlation_labels.log", level=DEBUG)

logger = getLogger("main")


def calculate_persons_cor_ci(
    s1: ndarray, s2: ndarray, alpha: int = 0.05
) -> tuple[float, tuple[float, float]]:
    # substitute nans with 0s
    s1[s1 != s1] = 0
    # s2[s2 != s2] = 0

    # Calculate Pearson's correlation coefficient and p-value
    correlation_coefficient, p_value = pearsonr(s1, s2)

    # Calculate confidence interval
    confidence_interval = pearsonr_ci(r=correlation_coefficient, n=len(s1), alpha=alpha)

    return float(correlation_coefficient), tuple(
        float(el) for el in confidence_interval
    )


def main():
    path_to_config: str = (
        "src/run/statistical_analysis/config_correlation_labels.yml"
    )

    logger.info("Starting model training")
    configs: dict[str, Any] = load_config(path=path_to_config)
    logger.debug("Configs loaded")

    path_to_features_data: str = configs["path_to_features_data"]
    path_to_save_data: str = configs["path_to_save_data"]
    # artifacts: bool = configs["artifacts"]
    alpha: int = configs.get("alpha", 0.05)
    components: list[str] = configs["components"]

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

    correlation_results = defaultdict(lambda: defaultdict(lambda: dict()))
    for side_name, data, labels in [
        ("left", features_left, labels_left),
        ("right", features_right, labels_right),
    ]:
        for num_component, component_name in enumerate(components):
            for i, feature in tqdm(
                enumerate(EDA_FEATURE_NAMES),
                desc=f"Feature progress for component {component_name}, class {side_name}",
                total=len(EDA_FEATURE_NAMES),
            ):
                correlation_results[side_name][component_name][
                    feature
                ] = calculate_persons_cor_ci(
                    s1=data[:, i, num_component],
                    s2=labels,
                    alpha=alpha,
                )

    # save dictionary to json file
    with open(path_to_save_data, "w") as f:
        dump(correlation_results, f)



if __name__ == "__main__":
    main()
