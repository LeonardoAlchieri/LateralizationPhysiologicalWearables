from numpy import datetime64, array, asarray, ndarray, mean
from numpy import isnan, nan, sign
from typing import Callable, Any
from time import time
from functools import wraps
from logging import getLogger
from sys import __stdout__
import sys
from tqdm.auto import tqdm
from os import devnull
from pandas import (
    DataFrame,
    Index,
    Timestamp,
    DatetimeIndex,
    IndexSlice,
    MultiIndex,
    date_range,
    to_datetime,
    Series,
    Timedelta,
)
from joblib import Parallel, delayed

logger = getLogger("utils")

# TODO: define this with a json somewhere
SESSIONS_GROUPINGS: dict[str, list[str]] = {
    "baseline": ["baseline_1", "baseline_2", "baseline_3", "baseline_4", "baseline_5"],
    "cognitive_load": ["cognitive_load"],
    "fake_laughter": ["fake"],
    "clapping_hands": ["clapping_hands"],
    "funny_videos": [
        "baby",
        "people_car",
        "stand_up_comedy",
        "himym",
        "cat",
        "penguins",
        "man",
        "bbt",
        "people_falling",
    ],
}

INTENSITIES_MAPPING: dict[str, float] = {"low": 0, "medium": 1, "high": 2}


# decorater used to block function printing to the console
def blockPrinting(func):
    """Method to decorate a function and block all prints in it.

    Credits to Fowler on Stackoverflow → https://stackoverflow.com/a/52605530/17422200
    """

    def func_wrapper(*args, **kwargs):
        # block all printing to the console
        sys.stdout = open(devnull, "w")
        # call the method in question
        value = func(*args, **kwargs)
        # enable all printing to the console
        sys.stdout = __stdout__
        # pass the return value of the method back
        return value

    return func_wrapper


def get_execution_time(func: Callable):
    """Simple method to output to logger the execution time for a method. The method will
    output in a reasonable unit of measure

    Args:
        func (Callable): function to be evaluated the time of
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time()
        result: Any = func(*args, **kwargs)  # save the result to a name
        compute_time: int = time() - start
        if compute_time < 0.1:
            logger.info(
                "Computation time for %s: %.2f ms"
                % (func.__name__, compute_time * 1000)
            )
        elif compute_time < 60:
            logger.info(
                "Computation time for %s: %.2f s" % (func.__name__, compute_time)
            )
        elif compute_time / 60 < 60:
            logger.info(
                "Computation time for %s: %.1f min"
                % (func.__name__, (compute_time) / 60)
            )
        else:
            logger.info(
                "Computation time for %s: %.1f h"
                % (func.__name__, (compute_time) / 3600)
            )
        return result  # return the name

    return wrapper


def make_timestamp_idx(dataframe: DataFrame, data_name: str) -> DataFrame:
    """A simple method to make the timestamp in a dataframe as the index for it.

    Parameters
    ----------
    dataframe : DataFrame
        dataframe to make timestamp index
    side : str
        side, 'left' or 'right', to work on
    data_name : str
        type of data, e.g. "ACC" or "EDA"
    individual_name : str | None, optional
        name of the individual to work on, e.g. 's034', by default None

    Returns
    -------
    DataFrame
        returns the dataframe with the new index
    """
    #     NOTE: tested the alterative with regex, i.e. split on "." and then cast to int directly
    #     and the time is similar
    # dataframe.attrs["sampling frequency"] = int(float(dataframe.columns[0][-1]))
    # dataframe.attrs["start timestamp [unixtime]"] = dataframe.columns[0][0]
    dataframe.attrs["start timestamp"] = to_datetime(
        dataframe.attrs["start timestamp [unixtime]"], unit="s", utc=True
    )
    dataframe.attrs["start timestamp"] = dataframe.attrs["start timestamp"].tz_convert(
        "Europe/Rome"
    )
    # if not data_name == "ACC":
    #     dataframe.columns = [data_name]
    # else:
    if data_name == "ACC":
        dataframe.columns = [f"{data_name}_{axis}" for axis in ["x", "y", "z"]]
    index_timestamps = date_range(
        start=dataframe.attrs["start timestamp"],
        periods=len(dataframe.index),
        freq=f"{1/dataframe.attrs['sampling frequency']*1000}ms",
    )

    dataframe.index = index_timestamps
    # tuples_for_multiindex: list[tuple[str, DatetimeIndex]] = [
    #     (individual_name, index_timestamp) for index_timestamp in index_timestamps
    # ]
    # dataframe.index = MultiIndex.from_tuples(
    #     tuples_for_multiindex, names=["participant", "timestamp"]
    # )
    # dataframe.columns = MultiIndex.from_tuples(
    #     [(side, current_col) for current_col in dataframe.columns]
    # )

    #         NOTE: needed, otherwise not json seriazable
    dataframe.attrs["start timestamp"] = str(dataframe.attrs["start timestamp"])
    return dataframe.sort_index()


def slice_user_over_experiment_time(
    user_data: DataFrame, experimento_info: DataFrame, slicing_col: str = "experiment"
) -> DataFrame:
    """Simple method to slice the user data over one of the phases of the experiment, e.g.
    all of it, funny video segment, etc.

    Parameters
    ----------
    user_data : DataFrame
        dataframe contraining the biometrics information, e.g. EDA, ACC etc., of a single
        user
    experimento_info : DataFrame
        experiment info table, which contains information regarding the start and end time
        for the `slicing_col` variable
    slicing_col: str
        identifies the name of column to consider for the slicing; defaults to 'experiment'

    Returns
    -------
    DataFrame
        return the user dataframe but with the row sliced along the start and end of
        the event identified by `slicing_col`
    """

    # It is assumed this method is given the dataframe of a single user w/ a multi-level index
    user: str = user_data.index.get_level_values(0).unique()[0]
    user_experiment_info_events: Index = experimento_info.loc[
        IndexSlice[user, :], :
    ].index.get_level_values(1)
    if slicing_col in user_experiment_info_events:
        times_to_slice_over: DataFrame = prepare_experiment_times(
            experimento_info=experimento_info, event=slicing_col
        )

        start: Timestamp | datetime64 = times_to_slice_over.loc[
            user_data.index.unique(level=0), "start"
        ].unique()[0]
        end: Timestamp | datetime64 = times_to_slice_over.loc[
            user_data.index.unique(level=0), "end"
        ].unique()[0]
        return user_data.loc[IndexSlice[:, start:end], :]
    else:
        return None


def prepare_experiment_times(
    experimento_info: DataFrame, event: str = "experiment"
) -> DataFrame:
    """Simple method to clean the dataframe with the experiment times, depending on which
    area of interest is needed.

    Parameters
    ----------
    experimento_info : DataFrame
        dataframe containing the raw experiment information, as provided in the USILaugh
        dataset
    event : str, optional
        name of the column to be considered, by default 'experiment'

    Returns
    -------
    DataFrame
        the method returns the dataframe cleaned
    """
    times_to_slice_over_experiment: DataFrame = experimento_info.loc[
        IndexSlice[:, event], ["start", "end"]
    ]
    times_to_slice_over_experiment.index = (
        times_to_slice_over_experiment.index.droplevel(1)
    )
    return times_to_slice_over_experiment


def change_orientation(x: DataFrame) -> DataFrame:
    """Simple method to change orientation of a dataframe with a
    multi-level index.

    Parameters
    ----------
    x : DataFrame
        dataframe to change orientation of; required to have a "left" and "right" column

    Returns
    -------
    DataFrame
        dataframe with columns as index and vice versa
    """
    users = x.index.get_level_values(1).unique()
    event = x.index.get_level_values(0).unique()
    x = array(x)
    x = DataFrame(x, index=users, columns=event).T
    return x


def calculate_mean_difference(x: DataFrame, use_abs: bool = False) -> ndarray:
    """Simple method to evaluate the mean of the differences between the left
    and right side for a given dataframe.

    Parameters
    ----------
    x : DataFrame
        dataframe over which to evaluate the difference; required to have
        a "left" and "right" column
    use_abs : bool, optional
        if True, the differences will be considered absolute, otherwise signed, by default False

    Returns
    -------
    ndarray
        returns the array of the the differences
    """
    data1: Series = x.loc[:, "left"]
    data2: Series = x.loc[:, "right"]
    data1: ndarray = asarray(data1)
    data2: ndarray = asarray(data2)
    if use_abs:
        diff: ndarray = abs(
            data1 - data2
        )  # Absolute Difference between data1 and data2
    else:
        diff: ndarray = data1 - data2  # Difference between data1 and data2
    md = mean(diff)
    return md


def get_cliff_bin(
    x: Series, dull: list[str] | None = None, raise_nan: bool = False
) -> str:
    """Method to get bins for the cliff delta values. They are considered
    with sign, and follow the suggestions by Vargha and Delaney (2000))

    Parameters
    ----------
    x : Series
        cliff delta data to bin over
    dull : list[str] | None, optional
        bins over which to separate; if None, the default by Vargha and
        Delaney (2020) will be used, by default None
    raise_nan : bool, optional
        if True, the method will fail if a Cliff delta is nan, by default False

    Returns
    -------
    str
        returns the description of the bin

    Raises
    ------
    ValueError
        if raise_nan is True and a Cliff delta is nan
    ValueError
        if the value does not check any dull list controls, but is also not nun
    """
    x_sign = sign(x)
    x = abs(x)
    if dull is None:
        dull: dict[str, str] = {
            "small": 0.11,
            "medium": 0.28,
            "large": 0.43,
        }  # effect sizes from (Vargha and Delaney (2000)) "negligible" for the rest=
    if x < dull["small"]:
        return 0 * x_sign
    elif dull["small"] <= x < dull["medium"]:
        return 1 * x_sign
    elif dull["medium"] <= x < dull["large"]:
        return 2 * x_sign
    elif x >= dull["large"]:
        return 3 * x_sign
    else:
        if isnan(x):
            if raise_nan:
                raise ValueError("NaN value")
            else:
                return nan
        else:
            raise ValueError(f"{x} is not in the dull range")

def correct_session_name(session_name: str) -> str:
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

def prepare_data_for_concatenation(
    data: Series, session_name: str, mode: int = 1
) -> Series:
    # NOTE: session_names may contain the time as well, to which we are not interested.
    # The session name is the name of the day the user woke up. As such, we
    # have to add a +1 day to the session name to get the correct date.

    if mode == 1:
        session_id_corrected = session_name
    elif mode == 2:
        # session_id_corrected = correct_session_name(session_name)
        # NOTE: I moved the session name correction before everything
        session_id_corrected = session_name
    else:
        raise ValueError(f"Mode not supported. Please use 1 or 2. Received {mode}")

    tuples_for_multiindex: list[tuple[str, DatetimeIndex]] = [
        (session_id_corrected, index_timestamp) for index_timestamp in data.index
    ]

    data.index = MultiIndex.from_tuples(
        tuples_for_multiindex, names=["session", "timestamp"]
    )
    return data


def segment_over_experiment_time(
    data_dict: dict[str, dict[str, dict[str, Series]]], experiment_time: DataFrame
) -> dict[str, dict[str, dict[str, Series]]]:
    """This method segments the data over the experiment time. This should
    be done before running any standardization or normalization, since it
    would help to avoid large artefacts.

    Parameters
    ----------
    data_dict : dict[str, dict[str, Series]]
        dictionary containing the data to segment
    experiment_time : DataFrame
        dataframe with users as rows, and two columns, called "start" and
        "end", respectively containing the start and end time of the experiment

    Returns
    -------
    dict[str, dict[str, Series]]
        returns the same structure of the input, but with the data segmented
    """

    def auxiliary_selecting_start_end(
        data: DataFrame,
        user_id: str,
        session_id: str,
        n_levels_idx: int,
        selection: str = "start",
    ) -> ndarray:
        if n_levels_idx == 1:
            return data.loc[user_id, selection]
        elif n_levels_idx == 2:
            return data.loc[IndexSlice[user_id, session_id], selection]
        else:
            raise ValueError(f"n_levels_idx not supported: {n_levels_idx}")

    n_levels_idx = experiment_time.index.nlevels

    return {
        side: {
            user: {
                session: user_data.loc[
                    auxiliary_selecting_start_end(
                        experiment_time, user, session, n_levels_idx, "start"
                    ) : auxiliary_selecting_start_end(
                        experiment_time, user, session, n_levels_idx, "end"
                    )
                ]
                for session, user_data in user_data.items()
            }
            for user, user_data in side_data.items()
        }
        for side, side_data in data_dict.items()
    }


def remove_empty_sessions(data_dict: dict[str, dict[str, dict[str, Series]]]):
    # NOTE: first remove the sessions, then remove the users who do not have any sessions

    data_dict_sessions_removed = {
        side: {
            user: {
                session: user_data
                for session, user_data in user_data.items()
                if not user_data.empty
            }
            for user, user_data in side_data.items()
        }
        for side, side_data in data_dict.items()
    }
    return {
        side: {
            user: {session: user_data for session, user_data in user_data.items()}
            for user, user_data in side_data.items()
            if len(side_data) > 0
        }
        for side, side_data in data_dict_sessions_removed.items()
    }


def parallel_iteration(func):
    """Method to decorate a function to run over the nested dictionary in parallel."""

    def func_wrapper(data, n_jobs, *args, **kwargs):
        # n_jobs = -1

        if n_jobs == 1:

            def super_func(
                func: Callable,
                session_data: DataFrame,
                side_name,
                user_name,
                session_name,
                *args,
                **kwargs,
            ):
                intermediate_res = func(
                    # func=func,
                    session_data,
                    side_name=side_name,
                    user_name=user_name,
                    session_name=session_name,
                    *args,
                    **kwargs,
                )
                if isinstance(intermediate_res, DataFrame):
                    # new_cols = intermediate_res.columns
                    intermediate_res.attrs = session_data.attrs
                    return intermediate_res
                else:
                #     new_cols = session_data.columns if isinstance(session_data, DataFrame) else None
                    raise NotImplementedError("Not implemented yet when the output is not a DataFrame")

                # return DataFrame(
                #     intermediate_res,
                #     index=session_data.index,
                #     columns=new_cols if isinstance(session_data, DataFrame) else None,
                # )

            results = {
                side: {
                    user: {
                        session_name: super_func(
                            func=func,
                            session_data=session_data,
                            side_name=side,
                            user_name=user,
                            session_name=session_name,
                            *args,
                            **kwargs,
                        )
                        for session_name, session_data in user_data.items()
                    }
                    for user, user_data in tqdm(
                        data[side].items(),
                        desc=f'Filtering for side "{side}"',
                        colour="green",
                    )
                }
                for side in data.keys()
            }
        elif n_jobs > 1 or n_jobs == -1:

            def super_func(
                func: Callable,
                session_data: DataFrame,
                side_name,
                user_name,
                session_name,
                *args,
                **kwargs,
            ):
                intermediate_res = func(
                    session_data,
                    side_name=side_name,
                    user_name=user_name,
                    session_name=session_name,
                    *args,
                    **kwargs,
                )
                if isinstance(intermediate_res, DataFrame):
                    # new_cols = intermediate_res.columns
                    intermediate_res.attrs = session_data.attrs
                    return (session_name, intermediate_res,)
                else:
                #     new_cols = session_data.columns if isinstance(session_data, DataFrame) else None
                    raise NotImplementedError("Not implemented yet when the output is not a DataFrame")

                # return (
                #     session_name,
                #     DataFrame(
                #         intermediate_res,
                #         index=session_data.index,
                #         columns=new_cols
                #         if isinstance(session_data, DataFrame)
                #         else None,
                #     ),
                # )

            results = {
                side: {
                    user: Parallel(n_jobs=n_jobs)(
                        delayed(super_func)(
                            func,
                            session_data=session_data,
                            side_name=side,
                            user_name=user,
                            session_name=session_name,
                            *args,
                            **kwargs,
                        )
                        for session_name, session_data in user_data.items()
                    )
                    for user, user_data in tqdm(
                        data[side].items(),
                        desc=f'Applying method {func.__name__} on side data "{side}"',
                        colour="green",
                    )
                }
                for side in data.keys()
            }

            results = {
                side: {
                    user: {
                        session_name: DataFrame(
                            session_data,
                            index=data[side][user][session_name].index,
                            columns=data[side][user][session_name].columns
                            if isinstance(data[side][user][session_name], DataFrame)
                            else None,
                        )
                        if isinstance(session_data, Series)
                        or isinstance(session_data, ndarray)
                        else session_data
                        for (session_name, session_data) in user_data
                    }
                    for user, user_data in results[side].items()
                }
                for side in results.keys()
            }
        else:
            raise ValueError(f'Invalid value for "n_jobs": got {n_jobs}')

        return results

    return func_wrapper
