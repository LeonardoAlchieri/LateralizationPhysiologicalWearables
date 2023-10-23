from collections import defaultdict
from glob import glob
from logging import getLogger
from typing import Any
from warnings import warn

from pandas import (
    DataFrame,
    IndexSlice,
    MultiIndex,
    Timestamp,
    Series,
    read_csv,
    read_parquet,
    to_datetime,
)
from tqdm.auto import tqdm
from yaml import safe_load as load_yaml

from src.utils import get_execution_time
from src.utils.misc import get_all_users, get_all_sessions

logger = getLogger("io")


@get_execution_time
def load_config(path: str) -> dict[str, Any]:
    """Simple method to load yaml configuration for a given script.

    Args:
        path (str): path to the yaml file

    Returns:
        Dict[str, Any]: the method returns a dictionary with the loaded configurations
    """
    with open(path, "r") as file:
        config_params = load_yaml(file)
    return config_params


def load_nested_parquet_data(
    path_to_main_folder: str, side: str | None = None, data_type: str | None = None
) -> (
    defaultdict[str, defaultdict[str, dict[str, Series]]]
    | defaultdict[str, dict[str, Series | DataFrame]]
):
    """Simple method to load multiple parquet files in a folder > subfolder structure.
    The main folder should be given, and then the method will crawl and load all of the
    parquet files inside the folder structure, which is expected as:
    ```
    <main_folder>/<side>/<data_type>/files.parquet
    ```
    where `<side>` is expected to be either 'left' or 'right'

    Parameters
    ----------
    path_to_main_folder : str
        path to the folder
    side : str | None, optional
        side for the folder structure; it must thus match it (usually 'left' or 'right'
        expected); by default None. If None, it will be assumed to get both the 'left'
        and 'right' side. Defaults to None.
    data_type : str | None, optional
        data type for the folder structure; it must thus match it (e.g. 'EDA'); by default None.

    Returns
    -------
    defaultdict[str, defaultdict[str, Series | DataFrame]]
        the method returns a loaded default dictionary, with a triple structure:
        ```
        {side: {data_type: {user: Series or DataFrame}}}
        ```
        where `Series` (or `DataFrame`) is a Series (DataFrame) associated with the user
    """
    # FIXME: change from Series to series, or remove one level of dictionary
    if side is None:
        logger.debug("No side provided. Assuming both sides")
        sides: list[str] = ["right", "left"]
    else:
        sides: list[str] = [side]

    all_data_as_dict: defaultdict[str, defaultdict[str, Series]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for chosen_side in sides:
        logger.info(f"Loading for side {chosen_side}")
        all_cleaned_paths: list[str] = glob(
            f"{path_to_main_folder}/{chosen_side}/*/*.parquet"
        )
        if data_type is not None:
            all_cleaned_paths = [
                path for path in all_cleaned_paths if data_type in path
            ]
        logger.debug(f"All files to be loaded: {all_cleaned_paths}")
        tricky_tags: list[str] = ["tags", "IBI", "Table"]

        for path in tqdm(all_cleaned_paths):
            # NOTE: this condition can be tested without casting to list, but it will be
            # slower, for some reason
            if any([tag not in path for tag in tricky_tags]):
                data_loaded: DataFrame = read_parquet(path)
                # NOTE: if only one column is present, return as Series, otherwise as DataFrame
                if len(data_loaded.columns) == 1:
                    all_data_as_dict[chosen_side][path.split("/")[-2]][
                        path.split("/")[-1].split(".")[0]
                    ] = data_loaded.iloc[:, 0]
                else:
                    all_data_as_dict[chosen_side][path.split("/")[-2]][
                        path.split("/")[-1].split(".")[0]
                    ] = data_loaded
                del data_loaded
            else:
                RuntimeWarning(f"Tags {tricky_tags} loading is not implemented yet.")

    if data_type is None:
        return all_data_as_dict
    else:
        return {
            side: inner_dict[data_type] for side, inner_dict in all_data_as_dict.items()
        }


# TODO: remove one folder level → can just use a Series w/o problems
def load_and_prepare_data(
    path_to_main_folder: str,
    side: str | None = None,
    data_type: str | None = None,
    mode: int = 1,
    device: str = "E4",
    input_format: str = "csv",
) -> (
    defaultdict[str, defaultdict[str, dict[str, Series]]]
    | defaultdict[str, dict[str, Series | DataFrame]]
):
    """Simple method to load multiple parquet files in a folder > subfolder structure.
    The main folder should be given, and then the method will crawl and load all of the
    parquet files inside the folder structure, which is expected as:
    ```
    <main_folder>/<side>/<data_type>/files.parquet
    ```
    where `<side>` is expected to be either 'left' or 'right'

    Parameters
    ----------
    path_to_main_folder : str
        path to the folder
    side : str | None, optional
        side for the folder structure; it must thus match it (usually 'left' or 'right'
        expected); by default None. If None, it will be assumed to get both the 'left'
        and 'right' side. Defaults to None.
    data_type : str | None, optional
        data type for the folder structure; it must thus match it (e.g. 'EDA'); by default None.

    Returns
    -------
    defaultdict[str, defaultdict[str, Series | DataFrame]]
        the method returns a loaded default dictionary, with a triple structure:
        ```
        {side: {data_type: {user: Series or DataFrame}}}
        ```
        where `Series` (or `DataFrame`) is a Series (DataFrame) associated with the user
    """
    # FIXME: change from Series to series, or remove one level of dictionary
    if side is None:
        logger.debug("No side provided. Assuming both sides")
        sides: list[str] = ["right", "left"]
    else:
        sides: list[str] = [side]

    all_data_as_dict: defaultdict[
        str, defaultdict[str, defaultdict[str, Series]]
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for chosen_side in sides:
        logger.info(f"Loading for side {chosen_side}")
        if mode == 1:
            path_current_side_data: list[str] = glob(
                f"{path_to_main_folder}/*/{chosen_side}/{data_type}.{input_format}"
            )
        elif mode == 2:
            path_current_side_data: list[str] = glob(
                f"{path_to_main_folder}/*/*/{device}/{chosen_side}/{data_type}_*.{input_format}"
            )
        else:
            raise ValueError(f"Mode not recognized. Got {mode} instead of 1 or 2.")
        logger.debug(f"All files to be loaded: {path_current_side_data}")
        tricky_tags: list[str] = ["tags", "IBI", "Table"]

        for path in tqdm(path_current_side_data, desc="Loading data"):
            # NOTE: this condition can be tested without casting to list, but it will be
            # slower, for some reason
            if not any([tag in path for tag in tricky_tags]):
                if input_format == "parquet":
                    data_loaded: DataFrame = read_parquet(path, header=[0, 1])
                elif input_format == "csv":
                    data_loaded: DataFrame = read_csv(path, header=[0, 1])
                data_loaded.attrs["sampling frequency"] = int(
                    float(data_loaded.columns[0][-1])
                )
                data_loaded.attrs["start timestamp [unixtime]"] = data_loaded.columns[
                    0
                ][0]
                # NOTE: if only one column is present, return as Series, otherwise as DataFrame
                if mode == 1:
                    current_data_name: str = path.split("/")[-1].split(".")[0]
                    current_user_name: str = path.split("/")[-3]
                    # FIXME: in mode 1, there is no session, since only one experiment was carried out
                    current_session_name: str = "experiment"
                elif mode == 2:
                    current_data_name: str = (
                        path.split("/")[-1].split(".")[0].split("_")[0]
                    )
                    current_session_name: str = (
                        path.split("/")[-1].split(".")[0].split("_")[-1]
                    )
                    current_user_name: str = path.split("/")[-5]

                if len(data_loaded.columns) == 1:
                    if data_type is None:
                        all_data_as_dict[chosen_side][current_user_name][
                            current_session_name
                        ][current_data_name] = data_loaded.iloc[:, 0]
                    else:
                        all_data_as_dict[chosen_side][current_user_name][
                            current_session_name
                        ] = data_loaded.iloc[:, 0]
                else:
                    if data_type is None:
                        all_data_as_dict[chosen_side][current_user_name][
                            current_session_name
                        ][current_data_name] = data_loaded
                    else:
                        all_data_as_dict[chosen_side][current_user_name][
                            current_session_name
                        ] = data_loaded
                del data_loaded
            else:
                NotImplementedError(
                    f"Tags {tricky_tags} loading is not implemented yet."
                )

    return all_data_as_dict


def read_experimentinfo(path: str, user: str, mode: int = 1) -> DataFrame:
    """Simple method to read the experiment info file for a given user.

    Parameters
    ----------
    path : str
        path to the csv file, usually of the type <user>/experiment_info.csv
    user : str
        user name, e.g. s099
    mode : int, optional
        mode for the loading, by default 1. If 1, it will be assumed the
        structure from USILaughs, otherwise from MWC2022.

    Returns
    -------
    DataFrame
        returns the dataframe loaded and with the correct multi-level index
    """
    df: DataFrame = read_csv(path, index_col=0)
    df.index = MultiIndex.from_tuples([(user, idx) for idx in df.index])
    return df


def save_data(
    data_to_save: DataFrame, filepath: str, save_format: str = "parquet"
) -> None:
    """Simple auxiliary method to save dataframe to a file, but can handle both parquet
    and csv using just a simple string call.

    Parameters
    ----------
    data_to_save : DataFrame
        dataframe to save
    filepath : str
        file to the dataframe, without extention, which will be added depending on the
        `save_format` given
    save_format : str, optional
        save format for the data, either 'csv' or 'parquet' at the moment, by default 'parquet'

    Raises
    ------
    ValueError
        if a non-supported save format is asked, the method will raise an error
    """
    logger.info("Saving data")
    logger.debug(f"Save format selected: {save_format}")
    if save_format == "parquet":
        data_to_save.to_parquet(f"{filepath}.parquet")
    elif save_format == "csv":
        data_to_save.to_csv(f"{filepath}.csv")
    else:
        raise ValueError(
            f'{save_format} is not a valid format. Accepted: "parquet" or "csv"'
        )
    logger.info("Data saved successfully")


def load_processed_data(
    path: str, file_format: str | None = None
) -> defaultdict[str, defaultdict[str, Series | DataFrame]]:
    """Load data from path.

    Parameters
    ----------
    path : str
        Path to the data.
    file_format : str | None, optional
        File format of the data, by default None

    Returns
    -------
    defaultdict[str, defaultdict[str, Series | DataFrame]]
        Dictionary of data.

    Raises
    ------
    ValueError
        If no data is found in the path.
    NotImplementedError
        If the pickle file format is not supported.
    NotImplementedError
        If the json file format is not supported.
    NotImplementedError
        If the hdf file format is not supported.
    NotImplementedError
        If the feather file format is not supported.
    ValueError
        If a value other than the ones described above is given.
    """
    paths: list[str] = glob(path)
    if len(paths) == 0:
        raise ValueError(
            f"No data found in path {path}. Please check \
            that it contains the data or the formatting is correct."
        )
    data: defaultdict[str, defaultdict[str, Series]] = defaultdict(
        lambda: defaultdict()
    )

    if file_format is None:
        file_format: str = path.split(".")[-1]

    for file in tqdm(paths, desc="Loading data"):
        side_name = file.split("/")[-3]
        user_name = file.split("/")[-1].split(".")[0]
        if file_format == "parquet":
            loaded_df = read_parquet(file)
            if not loaded_df.empty:
                data[side_name][user_name] = loaded_df
            else:
                warn(
                    f"The data loaded for side {side_name} and user {user_name} is empty",
                    RuntimeWarning,
                )
        elif file_format == "csv":
            loaded_df = read_csv(file, index_col=0, header=[0])
            if not loaded_df.empty:
                data[side_name][user_name] = loaded_df
            else:
                warn(
                    f"The data loaded for side {side_name} and user {user_name} is empty",
                    RuntimeWarning,
                )
        elif file_format == "pickle":
            raise NotImplementedError(f"File format {file_format} not implemented yet.")
        elif file_format == "json":
            raise NotImplementedError(f"File format {file_format} not implemented yet.")
        elif file_format == "hdf":
            raise NotImplementedError(f"File format {file_format} not implemented yet.")
        elif file_format == "feather":
            raise NotImplementedError(f"File format {file_format} not implemented yet.")
        else:
            raise ValueError(f"File format {file_format} not recognized.")

    return data


def read_experiment_info(path: str, mode: int = 1) -> DataFrame:
    if mode == 1:

        def move_event_to_columns(df):
            starts = {}
            ends = {}
            for event in df.index.get_level_values(1).unique():
                starts[f"start_{event}"] = df.loc[IndexSlice[:, event], "start"].values
                ends[f"end_{event}"] = df.loc[IndexSlice[:, event], "end"].values
            return DataFrame({**starts, **ends})

        experiment_info = read_csv(path, index_col=[0, 1])
        indexes_to_drop = [
            idx
            for idx in experiment_info.index
            if "baseline" not in idx[1] and "cognitive_load" not in idx[1]
        ]
        experiment_info = experiment_info.drop(indexes_to_drop, inplace=False)
        experiment_info = experiment_info.groupby(
            axis=0, level=0, group_keys=True
        ).apply(move_event_to_columns)
        experiment_info.index = experiment_info.index.droplevel(1)
        experiment_info = experiment_info.applymap(to_datetime)
        experiment_info = experiment_info.applymap(
            lambda x: x.tz_localize("Europe/Rome")
        )
    elif mode == 2:
        experiment_info = read_csv(path, index_col=[0, 1])
        experiment_info["actual_bed_time"] = experiment_info["actual_bed_time"].apply(
            to_datetime
        )
        experiment_info["wake_up_time"] = experiment_info["wake_up_time"].apply(
            to_datetime
        )
        experiment_info["actual_bed_time"] = experiment_info[
            "actual_bed_time"
        ].dt.tz_localize("Europe/Rome")
        experiment_info["wake_up_time"] = experiment_info[
            "wake_up_time"
        ].dt.tz_localize("Europe/Rome")
    else:
        raise ValueError(f"Mode {mode} not recognized.")

    return experiment_info


def filter_sleep_nights(
    data: defaultdict[str, defaultdict[str, Series | DataFrame]],
    experiment_info: DataFrame,
    format_data: str = "processed",
):
    users = get_all_users(data)

    if format_data == "processed":
        counter = 0
        for user in tqdm(sorted(users), desc="flitering user progress", colour="red"):
            sessions_all = get_all_sessions(
                user_data_left=data["left"][user], user_data_right=data["right"][user]
            )
            for session in sessions_all:
                if (user, session) not in experiment_info.index:
                    counter += 1
                    for side in data.keys():
                        sessions_to_keep = [
                            data_ses
                            for data_ses in data[side][user]
                            .index.get_level_values(0)
                            .unique()
                            if data_ses != session
                        ]
                        data[side][user] = data[side][user].loc[
                            IndexSlice[sessions_to_keep, :], :
                        ]
                else:
                    continue
        print(f"Removed {counter} sessions")
    elif format_data == "raw":
        counter = 0
        for user in tqdm(sorted(users), desc="flitering user progress", colour="red"):
            sessions_all = get_all_sessions(
                user_data_left=data["left"][user], user_data_right=data["right"][user]
            )
            for session in sessions_all:
                if (user, session) not in experiment_info.index:
                    counter += 1
                    for side in data.keys():
                        data[side][user] = {
                            key: val
                            for key, val in data[side][user].items()
                            if key != session
                        }
                else:
                    continue
        print(f"Removed {counter} sessions")
    else:
        raise ValueError(
            f"Format {format_data} not recognized. Accepted values are 'processed' or 'raw'"
        )
    return data


def get_closest_timestamp(t1: Timestamp, t2: Timestamp, how: str = "latest"):
    if how == "latest":
        return max(t1, t2)
    elif how == "earliest":
        return min(t1, t2)
    else:
        raise ValueError(f"how {how} not recognized")


def cut_to_shortest_series(
    data: defaultdict[str, defaultdict[str, Series | DataFrame]]
) -> defaultdict[str, defaultdict[str, Series | DataFrame]]:
    users = get_all_users(data)

    for user in tqdm(sorted(users), desc="flitering user progress", colour="red"):
        sessions_all = get_all_sessions(
            user_data_left=data["left"][user], user_data_right=data["right"][user]
        )
        for session in sessions_all:
            start_timestamp = get_closest_timestamp(
                t1=data["left"][user][session].index[0],
                t2=data["right"][user][session].index[0],
                how="latest",
            )

            end_timestamp = get_closest_timestamp(
                t1=data["left"][user][session].index[-1],
                t2=data["right"][user][session].index[-1],
                how="earliest",
            )

            data["left"][user][session] = data["left"][user][session].loc[
                start_timestamp:end_timestamp
            ]
            data["right"][user][session] = data["right"][user][session].loc[
                start_timestamp:end_timestamp
            ]
            
    return data