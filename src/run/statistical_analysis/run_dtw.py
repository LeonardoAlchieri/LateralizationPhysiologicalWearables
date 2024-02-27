####################################
####################################
###### NOTICE: THIS IS NOT USED AND THE RESULTS ARE ACTUALLY NOT CORRECTLY FORMATTED
###### THE RESULTS SHOULD BE SEPARATED BY LABEL (0 OR 1), BUT IT DOES NOT RIGHT NOW
####################################
####################################

from logging import DEBUG, INFO, basicConfig, getLogger
from sys import path
from json import dump as json_dump
from collections import defaultdict
from typing import Any
from warnings import warn
from joblib import Parallel, delayed

from fastdtw import fastdtw
from numpy import nan
from scipy.spatial.distance import euclidean
from tqdm.auto import tqdm

path.append(".")
from src.utils.io import load_config, load_processed_data
from src.utils.plots import plot_heatmap_boxplot
from src.utils.pre_processing import trim_to_shortest

basicConfig(filename="logs/run_dtw.log", level=DEBUG)

logger = getLogger("main")


def run_dtw_calculation_event_average(
    physiological_data: dict, signal_component: str, n_jobs: int = -1, **kwargs
) -> dict[str, dict[str, float]]:
    users_left = physiological_data["left"].keys()
    users_right = physiological_data["right"].keys()
    user_list = list(set(users_left) & set(users_right))

    # Define a function that takes in a user and computes the DTW distance
    def compute_dtw(user: str, physiological_data: dict, session_name: str):
        x = (
            physiological_data["right"][user][session_name]
            .loc[:, [signal_component]]
            .values.reshape(-1, 1)
        )
        y = (
            physiological_data["left"][user][session_name]
            .loc[:, [signal_component]]
            .values.reshape(-1, 1)
        )
        if len(x) == 0 or len(y) == 0:
            warn(f"Shape of x or y is 0 for {user}, {session_name}")
            return (
                user,
                session_name,
                nan,
            )
        return (
            user,
            session_name,
            fastdtw(x=x, y=y, dist=kwargs.get("distance", euclidean))[0] / x.shape[0],
        )

    # Use Parallel and delayed to run the function in parallel
    dtws = Parallel(n_jobs=n_jobs)(
        delayed(compute_dtw)(user, physiological_data, session_name)
        for user in tqdm(user_list, desc="User progress", colour="blue")
        for session_name in tqdm(
            list(
                set(physiological_data["right"][user].keys())
                & set(physiological_data["left"][user].keys())
            ),
            desc="Session progress",
            colour="green",
            disable=(
                len(
                    set(physiological_data["right"][user].keys())
                    & set(physiological_data["left"][user].keys())
                )
                < 3
            )
        )
    )

    # Convert the results to a dictionary
    new_dtws = defaultdict(dict)
    for user, session_name, dtw in dtws:
        new_dtws[user][session_name] = dtw

    return new_dtws


def main():
    path_to_config: str = "src/run/statistical_analysis/config_dtw.yml"

    logger.info("Starting model training")
    configs: dict[str, Any] = load_config(path=path_to_config)
    logger.debug("Configs loaded")

    path_to_data: str = configs["path_to_data"]
    data_format: str = configs["data_format"]
    signal_component: str = configs["signal_component"]
    path_to_save_data: str = configs["path_to_save_data"]
    n_jobs: int = configs["n_jobs"]

    physiological_data = load_processed_data(path=path_to_data, file_format=data_format)

    physiological_data = trim_to_shortest(signal_data=physiological_data)

    dtws = run_dtw_calculation_event_average(
        physiological_data=physiological_data, n_jobs=n_jobs, signal_component=signal_component
    )
    
    # save results to json
    with open(path_to_save_data, "w") as f:
        json_dump(dtws, f)


if __name__ == "__main__":
    main()
