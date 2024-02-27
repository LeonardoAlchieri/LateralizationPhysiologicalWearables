from collections import defaultdict
from json import dump as json_dump
from logging import DEBUG, INFO, basicConfig, getLogger
from sys import path
from typing import Any, Callable
from warnings import warn

from arch.unitroot import ADF, DFGLS, KPSS, PhillipsPerron, ZivotAndrews
from matplotlib.pyplot import figure, savefig, show, title
from pandas import DataFrame, IndexSlice, Timestamp
from seaborn import heatmap
from tqdm.auto import tqdm

path.append(".")
from src.utils.io import load_config, load_processed_data
from src.utils.pre_processing import trim_to_shortest

basicConfig(filename="logs/run_stationarity_tests.log", level=DEBUG)

logger = getLogger("main")

possible_methods: dict[str, Callable] = {
    "ADF": ADF,
    "DFGLS": DFGLS,
    "KPSS": KPSS,
    "PhillipsPerron": PhillipsPerron,
    "ZivotAndrews": ZivotAndrews,
}


def check_stationarity(
    physiological_data: dict[str, defaultdict[str, defaultdict[str, DataFrame]]],
    data_source: str,
    signal_component: str,
    stationarity_method: Callable = KPSS,
    invert_pvalues: bool = False,
    nested: bool = False,
    no_plot: bool = False,
):
    users_left = physiological_data["left"].keys()
    users_right = physiological_data["right"].keys()
    user_list = list(set(users_left) & set(users_right))

    statistical_test = {
        side: {
            user: {
                session_name: (
                    stationarity_method(
                        physiological_data[side][user][session_name]
                        .loc[:, [signal_component]]
                        .values
                    ).pvalue
                    if not invert_pvalues
                    else 1
                    - stationarity_method(
                        physiological_data[side][user][session_name]
                        .loc[:, signal_component]
                        .values
                    ).pvalue
                )
                if (
                    physiological_data[side][user][session_name]
                    .iloc[:, 0]
                    .values.shape[0]
                    > 0
                )
                else None
                for session_name in list(
                    set(physiological_data["right"][user].keys())
                    & set(physiological_data["left"][user].keys())
                )
            }
            for user in tqdm(user_list, desc="User progress", colour="blue")
        }
        for side in ["left", "right"]
    }
    if nested:
        reform = {
            (outerKey, innerKey): values
            for outerKey, innerDict in statistical_test.items()
            for innerKey, values in innerDict.items()
        }
        df_to_save = DataFrame.from_dict(reform).stack(level=1, dropna=False).T
    else:
        df_to_save: DataFrame = DataFrame(statistical_test).sort_index().T
    if not no_plot:
        figure(figsize=(len(df_to_save.columns), len(df_to_save)))
        heatmap(
            df_to_save,
            xticklabels=df_to_save.columns,
            vmax=1,
            vmin=0,
            center=0,
            cmap="coolwarm",
            yticklabels=df_to_save.index,
            annot=True,
        )
        title(f"P-values of {stationarity_method.__name__} test")
        savefig(
            f"../visualizations/{stationarity_method.__name__}_pvalues_{data_source}.pdf",
            bbox_inches="tight",
        )
        show()
        return df_to_save
    else:
        return statistical_test


from pandas import DatetimeIndex, MultiIndex, Timedelta, to_datetime


def correct_session_idx(session_name: str):
    session_time: str = session_name.split("-")[1]
    if session_time[0] == "0":
        session_id_corrected: Timestamp = to_datetime(
            session_name.split("-")[0], format="%y%m%d"
        )
    else:
        session_id_corrected: Timestamp = to_datetime(
            session_name.split("-")[0], format="%y%m%d"
        ) + Timedelta("1D")
    session_id_corrected: str = str(session_id_corrected.date())
    return session_id_corrected


def super_correct_session_idx(idx):
    correct = [correct_session_idx(el) for el in idx]
    if len(correct) > len(set(correct)):
        new_correct = []
        i = 0
        for el in correct:
            if el in new_correct:
                new_correct.append(el + f"_{i}")
                i += 1
            else:
                new_correct.append(el)
                i = 0
        correct = new_correct
    return correct


def correct_session_names(df: DataFrame):
    df.index = df.index.set_levels(
        super_correct_session_idx(df.index.levels[0]), level=0
    )
    return df


def main():
    path_to_config: str = "src/run/statistical_analysis/config_stationarity_tests.yml"

    logger.info("Starting model training")
    configs: dict[str, Any] = load_config(path=path_to_config)
    logger.debug("Configs loaded")

    # n_workers: int = configs["n_workers"]
    dataset: str = configs["dataset"]
    path_to_data: str = configs["path_to_data"]
    data_format: str = configs["data_format"]
    signal_component: str = configs["signal_component"]
    methods: list[str] | None = configs["methods"]
    path_to_save_data: str = configs["path_to_save_data"]

    physiological_data = load_processed_data(path=path_to_data, file_format=data_format)

    physiological_data = trim_to_shortest(signal_data=physiological_data)

    if methods is None:
        warn(
            "Methods to run stationarity not specified. Runnining all — might be heavy computationally."
        )
        methods: list[Callable] = [ADF, DFGLS, KPSS, PhillipsPerron, ZivotAndrews]
    else:
        methods: list[Callable] = [possible_methods[method] for method in methods]

    test_result: dict[str, dict[str, dict[Any, Any | None]]] = dict()
    for method in tqdm(methods, desc="Methods progress", colour="green"):
        test_result[method.__name__] = check_stationarity(
            physiological_data=physiological_data,
            data_source=dataset,
            signal_component=signal_component,
            invert_pvalues=False,
            nested=True,
            stationarity_method=method,
            no_plot=True,
        )

    # save results to json
    with open(path_to_save_data, "w") as f:
        json_dump(test_result, f)


if __name__ == "__main__":
    main()
