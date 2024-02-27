from logging import DEBUG, INFO, basicConfig, getLogger
from sys import path
from json import dump as json_dump
from typing import Any
from warnings import warn

from joblib import Parallel, delayed
from numpy import nan, ndarray
from pandas import DataFrame
from tqdm.notebook import tqdm

path.append(".")
from src.utils.dcca import detrended_correlation
from src.utils.dcca.cross_correlation import cross_correlation
from src.utils.io import load_config, load_processed_data
from src.utils.plots import plot_heatmap_boxplot
from src.utils.pre_processing import trim_to_shortest

basicConfig(filename="logs/run_stationarity_tests.log", level=DEBUG)

logger = getLogger("main")


def time_lagged_cross_correlation_per_user(
    right_side_data: ndarray,
    left_side_data: ndarray,
    time_scale: int | None = None,
    max_time_lag: int | None = None,
    detrended: bool = True,
    n_jobs: int = -1,
    **kwargs,
) -> list[float]:
    if max_time_lag is None:
        max_time_lag = len(right_side_data)
    # TODO: implement negative time lags

    if right_side_data.shape[0] == 0 or left_side_data.shape[0] == 0:
        warn(f'No data for current input {kwargs.get("progress_item_name", None)}')
        return []
    time_lags = range(0, max_time_lag)

    if detrended:
        if time_scale is None:
            raise ValueError(
                f"time_scale must be provided for detrended correlation. Given {time_scale}"
            )
        try:
            result = Parallel(n_jobs=n_jobs, backend="threading")(
                delayed(detrended_correlation)(
                    x=right_side_data,
                    y=left_side_data,
                    time_scale=time_scale,
                    time_lag=time_lag,
                )
                for time_lag in tqdm(
                    time_lags,
                    desc=f'Calculating detrended cross correlation {kwargs.get("progress_item_name", None)}',
                    colour=kwargs.get("tqdm_color", "green"),
                )
            )
        except ValueError as e:
            print(
                f"{right_side_data.shape=}, {left_side_data.shape=}, {time_scale=}, {time_lags=}"
            )
            raise e
        return result
    else:
        warn("Calculation is not parallel. n_jobs ignored.")
        return [
            cross_correlation(x=right_side_data, y=left_side_data, time_lag=time_lag)
            for time_lag in tqdm(
                time_lags,
                desc=f'Calculating cross correlation {kwargs.get("progress_item_name", None)}',
                colour=kwargs.get("tqdm_color", "green"),
            )
        ]


def perform_correlation(
    physiological_data: dict,
    data_name: str,
    time_scale: int | None = None,
    max_time_lag: int = 200,
    detrended: bool = True,
    measure_name: str = "max dcca",
    n_jobs: int = -1,
):
    users_left = physiological_data["left"].keys()
    users_right = physiological_data["right"].keys()
    user_list = list(set(users_left) & set(users_right))

    dccas = {
        user: {
            session_name: time_lagged_cross_correlation_per_user(
                right_side_data=physiological_data["right"][user][session_name]
                .iloc[:, 0]
                .values,
                left_side_data=physiological_data["left"][user][session_name]
                .iloc[:, 0]
                .values,
                time_scale=time_scale,
                detrended=detrended,
                max_time_lag=max_time_lag,
                n_jobs=n_jobs,
                progress_item_name=f"{user}_{session_name}",
            )
            for session_name in list(
                set(physiological_data["right"][user].keys())
                & set(physiological_data["left"][user].keys())
            )
        }
        for user in tqdm(user_list, desc="User progress", colour="blue")
    }

    max_dccas = {
        user: max(dcca) if len(dcca) > 0 else nan for user, dcca in dccas.items()
    }
    try:
        plot_heatmap_boxplot(
            data=max_dccas,
            measure_name=measure_name,
            nested=True,
            vmax=1,
            vmin=-1,
            center=0,
            data_name=data_name,
        )
        return dccas
    except:
        return dccas


def main():
    
    path_to_config: str = "src/run/statistical_analysis/config_dcca.yml"

    logger.info("Starting model training")
    configs: dict[str, Any] = load_config(path=path_to_config)
    logger.debug("Configs loaded")
    
    path_to_data: str = configs["path_to_data"]
    data_format: str = configs["data_format"]
    path_to_save_data: str = configs["path_to_save_data"]
    time_scale: int = configs["time_scale"]
    detrended: bool = configs.get("detrended", True)

    physiological_data = load_processed_data(path=path_to_data, file_format=data_format)

    physiological_data = trim_to_shortest(signal_data=physiological_data)

    dcca = perform_correlation(
        physiological_data=physiological_data,
        time_scale=time_scale,
        detrended=detrended,
        max_time_lag=32 * 2,
        measure_name="max dcca",
        data_name="mwc2022",
        n_jobs=-1,
    )
    
    # save results to json
    with open(path_to_save_data, "w") as f:
        json_dump(dcca, f)


if __name__ == "__main__":
    main()
