from typing import Any
from glob import glob
from logging import getLogger
from tqdm import tqdm
from collections import defaultdict
from yaml import safe_load as load_yaml
from pandas import DataFrame, MultiIndex, Series, concat, read_csv, read_parquet

from src.utils import get_execution_time, make_timestamp_idx

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
) -> defaultdict[str, defaultdict[str, dict[str, Series]]] | defaultdict[
    str, dict[str, Series | DataFrame]
]:
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
) -> defaultdict[str, defaultdict[str, dict[str, Series]]] | defaultdict[
    str, dict[str, Series | DataFrame]
]:
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
                f"{path_to_main_folder}/*/{chosen_side}/{data_type}.csv"
            )
        elif mode == 2:
            path_current_side_data: list[str] = glob(
                f"{path_to_main_folder}/*/*/{device}/{chosen_side}/{data_type}_*.csv"
            )
        else:
            raise ValueError(f"Mode not recognized. Got {mode} instead of 1 or 2.")
        logger.debug(f"All files to be loaded: {path_current_side_data}")
        tricky_tags: list[str] = ["tags", "IBI", "Table"]

        for path in tqdm(path_current_side_data, desc="Loading data"):
            # NOTE: this condition can be tested without casting to list, but it will be
            # slower, for some reason
            if not any([tag in path for tag in tricky_tags]):
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
                        ] = data_loaded.iloc[:, 0]
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
