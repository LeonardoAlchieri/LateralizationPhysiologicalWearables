from collections import defaultdict
from itertools import combinations
from logging import DEBUG, INFO, basicConfig, getLogger
from sys import path
from typing import Any
from json import dump

from mpsci.stats import pearsonr, pearsonr_ci
from scipy.stats import pearsonr as pearsonr_scipy
from numpy import load, ndarray, isnan
from tqdm.auto import tqdm
from dcca.cross_correlation import cross_correlation

path.append(".")
from src.feature_extraction.eda import EDA_FEATURE_NAMES
from src.utils.io import load_config

basicConfig(filename="logs/run_correlation_rl_features.log", level=DEBUG)

logger = getLogger("main")


def calculate_persons_cor_ci(
    s1: ndarray, s2: ndarray, alpha: float = 0.05
) -> tuple[float, tuple[float, float]]:
    # substitute nans with 0s
    s1[s1 != s1] = 0
    s2[s2 != s2] = 0

    # Calculate Pearson's correlation coefficient and p-value
    try:
        correlation_coefficient, p_value = pearsonr(s1, s2)
    except ValueError:
        correlation_coefficient, p_value = pearsonr_scipy(x=s1, y=s2, alternative="two-sided")

    if isnan(float(correlation_coefficient)):
        print('here')
    # Calculate confidence interval
    confidence_interval = pearsonr_ci(r=correlation_coefficient, n=len(s1), alpha=alpha)

    return (
        float(correlation_coefficient),
        float(p_value),
        tuple(float(el) for el in confidence_interval),
    )


def main():
    path_to_config: str = (
        "src/run/statistical_analysis/config_correlation_rl_features.yml"
    )

    logger.info("Starting model training")
    configs: dict[str, Any] = load_config(path=path_to_config)
    logger.debug("Configs loaded")

    path_to_features_data: str = configs["path_to_features_data"]
    path_to_save_data: str = configs["path_to_save_data"]
    # artifacts: bool = configs["artifacts"]
    alpha: int = configs.get("alpha", 0.05)
    components: list[str] = configs["components"]
    bilateral: bool = configs["bilateral"]

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
    if bilateral:
        features_diff = data["features_diff"]
        labels_diff = data["labels_diff"]
        positive_data_diff = features_diff[labels_diff == 1]
        negative_data_diff = features_diff[labels_diff == 0]

    correlation_results = defaultdict(lambda: defaultdict(lambda: dict()))
    class_comparisons = {
        "positive": (
            combinations(
                [
                    (positive_data_left, "left"),
                    (positive_data_right, "right"),
                    (positive_data_diff, "diff"),
                ],
                2,
            )
            if bilateral
            else combinations(
                [(positive_data_left, "left"), (positive_data_right, "right")], 2
            )
        ),
        "negative": (
            combinations(
                [
                    (negative_data_left, "left"),
                    (negative_data_right, "right"),
                    (negative_data_diff, "diff"),
                ],
                2,
            )
            if bilateral
            else combinations(
                [(negative_data_left, "left"), (negative_data_right, "right  ")], 2
            )
        ),
    }
    for class_name, data_combinations in tqdm(class_comparisons.items(), desc="Class progress", position=0, leave=True):
        for first_side, second_side in data_combinations:
            for num_component, component_name in enumerate(components):
                for i, feature in tqdm(
                    enumerate(EDA_FEATURE_NAMES),
                    desc=f"Feature progress for component {component_name}, class {class_name}",
                    total=len(EDA_FEATURE_NAMES),
                    leave=False,
                    position=1,
                ):
                    data_first, data_second = first_side[0], second_side[0]
                    label_first, label_second = first_side[1], second_side[1]
                    correlation_results[f"{class_name}-{label_first}/{label_second}"][
                        component_name
                    ][feature] = calculate_persons_cor_ci(
                        s1=data_first[:, i, num_component],
                        s2=data_second[:, i, num_component],
                        alpha=alpha,
                    )

    # save dictionary to json file
    with open(path_to_save_data, "w") as f:
        dump(correlation_results, f)


if __name__ == "__main__":
    main()
