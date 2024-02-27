from collections import defaultdict
from json import dump as json_dump
from logging import DEBUG, INFO, basicConfig, getLogger
from sys import path
from typing import Any, Callable
from warnings import warn

from arch.unitroot import ADF, DFGLS, KPSS, PhillipsPerron, ZivotAndrews
from matplotlib.pyplot import figure, savefig, show, title
from pandarallel import pandarallel
from pandas import DataFrame, IndexSlice, Timestamp
from seaborn import heatmap
from tqdm.auto import tqdm

path.append(".")
from src.utils.io import load_config, load_processed_data

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
            user: (
                physiological_data[side][user]
                .iloc[:, [0]]
                .groupby(axis=0, level=0)
                .parallel_apply(lambda x: stationarity_method(x.values).pvalue)
                .to_dict()
                if not invert_pvalues
                else physiological_data[side][user]
                .iloc[:, [0]]
                .groupby(axis=0, level=0)
                .parallel_apply(lambda x: 1 - stationarity_method(x.values).pvalue)
                .to_dict()
            )
            if physiological_data[side][user].iloc[:, 0].values.shape[0] > 0
            else None
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


def trim_to_shortest(
    signal_data: dict[str, defaultdict[str, defaultdict[str, DataFrame]]],
) -> dict[str, defaultdict[str, defaultdict[str, DataFrame]]]:
    """This method trims the physiological data to the shortest session length,
    between data from the left and right side of the body. Necessary only for the
    BiHeartS dataset, since for USILaughs the sessions match perfectly.

    Parameters
    ----------
    signal_data : dict[str, defaultdict[str, defaultdict[str, DataFrame]]]
        data to be trimmed

    Returns
    -------
    dict[str, defaultdict[str, defaultdict[str, DataFrame]]]
        same as input data for the format, but with each timeseries trimmed to the
        shortest length of the two sides

    Raises
    ------
    RuntimeError
        _description_
    RuntimeError
        _description_
    """
    new_physiological_data = defaultdict(lambda: defaultdict(lambda: defaultdict()))

    users_left = list(signal_data["left"].keys())
    users_right = list(signal_data["right"].keys())
    users = list(set(users_left) & set(users_right))
    for user in tqdm(users, desc=f"Trimming. User progress"):
        user_data_left: DataFrame = signal_data["left"][user]
        user_data_right: DataFrame = signal_data["right"][user]
        sessions_left = user_data_left.index.get_level_values("session").unique()
        sessions_right = user_data_right.index.get_level_values("session").unique()
        sessions = list(set(sessions_left) & set(sessions_right))
        for session in sessions:
            session_data_left = user_data_left.loc[IndexSlice[session, :], :].droplevel(
                level=0, axis=0
            )
            session_data_right = user_data_right.loc[
                IndexSlice[session, :], :
            ].droplevel(level=0, axis=0)

            max_start = max(
                session_data_left.index.get_level_values("timestamp").min(),
                session_data_right.index.get_level_values("timestamp").min(),
            )
            min_end = min(
                session_data_left.index.get_level_values("timestamp").max(),
                session_data_right.index.get_level_values("timestamp").max(),
            )

            session_data_left = session_data_left[
                (session_data_left.index >= max_start)
                & (session_data_left.index <= min_end)
            ]
            session_data_right = session_data_right[
                (session_data_right.index >= max_start)
                & (session_data_right.index <= min_end)
            ]
            if session_data_left.shape != session_data_right.shape:
                longest_index = (
                    session_data_left.index
                    if session_data_left.shape[0] > session_data_right.shape[0]
                    else session_data_right.index
                )
                shortest_index = (
                    session_data_right.index
                    if session_data_left.shape[0] > session_data_right.shape[0]
                    else session_data_left.index
                )
                for el in longest_index:
                    if el not in shortest_index:
                        cutoff_index = el
                        break
                session_data_left = session_data_left[
                    (session_data_left.index < cutoff_index)
                ]
                session_data_right = session_data_right[
                    (session_data_right.index < cutoff_index)
                ]
                if session_data_left.shape != session_data_right.shape:
                    raise RuntimeError("fuck")

            if len(session_data_left) == 0:
                raise RuntimeError("double fuck")
            new_physiological_data["left"][user][session] = session_data_left
            new_physiological_data["right"][user][session] = session_data_right

    return new_physiological_data


def main():
    path_to_config: str = "src/run/statistical_analysis/config_stationarity_tests.yml"

    logger.info("Starting model training")
    configs: dict[str, Any] = load_config(path=path_to_config)
    logger.debug("Configs loaded")

    n_workers: int = configs["n_workers"]
    dataset: str = configs["dataset"]
    path_to_data: str = configs["path_to_data"]
    data_format: str = configs["data_format"]
    methods: list[str] | None = configs["methods"]
    path_to_save_data: str = configs["path_to_save_data"]

    pandarallel.initialize(progress_bar=False, nb_workers=n_workers)

    physiological_data = load_processed_data(path=path_to_data, file_format=data_format)

    if methods is None:
        warn(
            "Methods to run stationarity not specified. Runnining all — might be heavy computationally."
        )
        methods: list[Callable] = [ADF, DFGLS, KPSS, PhillipsPerron, ZivotAndrews]
    else:
        methods: list[Callable] = [possible_methods[method] for method in methods]

    test_result: dict[str, dict[str, dict[Any, Any | None]]] = dict()
    for method in tqdm(methods, desc='Methods progress', colour='green'):
        test_result[method.__name__] = check_stationarity(
            physiological_data=physiological_data,
            data_source=dataset,
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
