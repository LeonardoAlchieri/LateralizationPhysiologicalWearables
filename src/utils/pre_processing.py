# from joblib import Parallel, delayed
from collections import defaultdict
from logging import getLogger
from typing import Callable

from joblib import Parallel, delayed
from numpy import array, log1p, nanmean, nanstd, ndarray, stack
from pandas import DataFrame, IndexSlice, Series, concat
from sklearn.preprocessing import (PowerTransformer, QuantileTransformer,
                                   RobustScaler)
from tqdm.auto import tqdm

from src.utils import prepare_data_for_concatenation

logger = getLogger("pre_processing")


def concate_session_data(
    data_dict: defaultdict[str, dict[str, dict[str, Series]]], n_jobs: int = 1
) -> dict[str, dict[str, Series]]:
    """Concatenate data from different sessions for each user.

    Args:
        data_dict (defaultdict[str, dict[str, dict[str, Series]]]): [description]

    Returns:
        dict[str, dict[str, Series]]: [description]
    """
    if n_jobs == 1:
        data_dict: defaultdict[str, dict[str, Series]] = {
            side: {
                user: concat(
                    [
                        prepare_data_for_concatenation(
                            data=data_dict[side][user][session], session_name=session
                        )
                        for session in data_dict[side][user].keys()
                    ],
                    axis=0,
                    join="outer",
                ).sort_index()
                for user in tqdm(
                    data_dict[side].keys(),
                    desc=f"Concatenating user data for side {side}",
                    colour="green",
                )
                if len(data_dict[side][user]) > 0
            }
            for side in data_dict.keys()
        }
    elif n_jobs > 1 or n_jobs == -1:
        data_dict: defaultdict[str, dict[str, Series]] = {
            side: {
                user: concat(
                    Parallel(n_jobs=n_jobs)(
                        delayed(prepare_data_for_concatenation)(
                            data=data_dict[side][user][session], session_name=session
                        )
                        for session in data_dict[side][user].keys()
                    ),
                    axis=0,
                    join="outer",
                ).sort_index()
                for user in tqdm(
                    data_dict[side].keys(),
                    desc=f"Concatenating user data for side {side}",
                    colour="green",
                )
                if len(data_dict[side][user]) > 0
            }
            for side in data_dict.keys()
        }
    else:
        raise ValueError(f"Invalid value for n_jobs: {n_jobs} (must be >= 1 or -1)")
    return data_dict


def rescaling(
    data: defaultdict[str, dict[str, dict[str, Series]]],
    rescaling_method: Callable,
    n_jobs: int = 1,
) -> defaultdict[str, dict[str, dict[str, Series]]]:
    """Rescale data using the specified method.

    Args:
        data (defaultdict[str, dict[str, dict[str, Series]]]): data to rescale
        rescaling_method (Callable): method to use for rescaling

    Returns:
        defaultdict[str, dict[str, dict[str, Series]]]: rescaled data
    """
    # TODO: do this using a decorator function
    if n_jobs == 1:
        data: defaultdict[str, dict[str, DataFrame]] = {
            side: {
                user: {
                    session: DataFrame(
                        rescaling_method(data[side][user][session])
                        if isinstance(data[side][user][session], Series)
                        or (
                            isinstance(data[side][user][session], DataFrame)
                            and len(data[side][user][session].columns) == 1
                        )
                        else stack(
                            [
                                rescaling_method(data[side][user][session].iloc[:, 0]),
                                data[side][user][session].iloc[:, 1].values,
                            ],
                            axis=1,
                        ),
                        index=data[side][user][session].index,
                        columns=data[side][user][session].columns
                        if isinstance(data[side][user][session], DataFrame)
                        else None,
                    )
                    for session in data[side][user].keys()
                }
                for user in tqdm(
                    data[side].keys(),
                    desc=f"Rescaling data for side {side}",
                    colour="blue",
                )
            }
            for side in data.keys()
        }
    elif n_jobs > 1 or n_jobs == -1:

        def support_rescaling(session: str, session_data: DataFrame):
            return (
                session,
                DataFrame(
                    rescaling_method(session_data.iloc[:, 0].values),
                    index=session_data.index,
                    columns=session_data.columns,
                ),
            )

        data = {
            side: {
                user: Parallel(n_jobs=n_jobs)(
                    delayed(support_rescaling)(session, data[side][user][session])
                    for session in data[side][user].keys()
                )
                for user in tqdm(
                    data[side].keys(),
                    desc=f"Rescaling data for side {side}",
                    colour="blue",
                )
            }
            for side in data.keys()
        }
        data = {
            side: {
                user: {
                    session_name: session_data
                    for (session_name, session_data) in user_acct_data
                }
                for user, user_acct_data in data[side].items()
            }
            for side in data.keys()
        }
    else:
        raise ValueError(f"Invalid value for n_jobs: {n_jobs} (must be >= 1 or -1)")
    return data


# TODO: probably remove and use some third party library
def standardize(signal: Series | ndarray | list) -> ndarray:
    """Simple method to standardize an EDA signal.

    Parameters
    ----------
    signal : Series | ndarray | list
        signal to standardize

    Returns
    -------
    ndarray
        returns an array standardized
    """
    y: ndarray = array((signal))

    yn: ndarray = (y - nanmean(y)) / nanstd(y)
    return yn


def min_max_normalization(signal: Series | ndarray | list) -> ndarray:
    """Simple method to normalize an EDA signal.

    Parameters
    ----------
    signal : Series | ndarray | list
        signal to normalize

    Returns
    -------
    ndarray
        returns an array normalized
    """
    y: ndarray = array((signal))

    yn: ndarray = (y - y.min()) / (y.max() - y.min())
    return yn


def robust_scaling_with_irq(signal: Series | ndarray | list) -> ndarray:
    """
    Perform Robust Scaling with Interquartile Range (IRQ) on the input data.

    Parameters:
        data (ndarray): Input data of shape (N,).

    Returns:
        ndarray: Scaled data of shape (N,) using Robust Scaling with IRQ.
    """
    y: ndarray = array((signal))
    scaler = RobustScaler()
    return scaler.fit_transform(y.reshape(-1, 1)).flatten()


def log_transformation(signal: Series | ndarray | list) -> ndarray:
    """
    Perform Log Transformation on the input data.

    Parameters:
        data (ndarray): Input data of shape (N,).

    Returns:
        ndarray: Transformed data of shape (N,) using Log Transformation.
    """
    y: ndarray = array((signal))
    return log1p(y)


def yeo_johnson_transformation(signal: Series | ndarray | list) -> ndarray:
    """
    Perform Yeo-Johnson Transformation on the input data.

    Parameters:
        data (ndarray): Input data of shape (N,).

    Returns:
        ndarray: Transformed data of shape (N,) using Yeo-Johnson Transformation.
    """
    y: ndarray = array((signal))
    transformer = PowerTransformer(method="yeo-johnson")
    return transformer.fit_transform(y.reshape(-1, 1)).flatten()


def quantile_transformation(signal: Series | ndarray | list) -> ndarray:
    """
    Perform Quantile Transformation on the input data.

    Parameters:
        data (ndarray): Input data of shape (N,).

    Returns:
        ndarray: Transformed data of shape (N,) using Quantile Transformation.
    """
    y: ndarray = array((signal))
    transformer = QuantileTransformer(output_distribution="uniform")
    return transformer.fit_transform(y.reshape(-1, 1)).flatten()


def get_rescaling_technique(rescaling_name: str) -> Callable:
    """Get the rescaling method from its name.

    Args:
        rescaling_name (str): name of the rescaling method

    Returns:
        Callable: rescaling method
    """
    if rescaling_name == "standardize":
        return standardize
    elif rescaling_name == "min_max":
        return min_max_normalization
    elif rescaling_name == "none":
        return lambda x: x
    elif rescaling_name == "robust_scaling_with_irq":
        return robust_scaling_with_irq
    elif rescaling_name == "log_transformation":
        return log_transformation
    elif rescaling_name == "yeo_johnson_transformation":
        return yeo_johnson_transformation
    elif rescaling_name == "quantile_transformation":
        return quantile_transformation
    else:
        raise ValueError(f"Invalid rescaling method: {rescaling_name}")


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
