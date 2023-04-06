from logging import INFO, basicConfig, getLogger
from os.path import basename

from joblib import Parallel, delayed
from numpy import isnan, ndarray, savetxt, stack
from pandas import DataFrame, IndexSlice, Series, Timedelta, Timestamp
from tqdm.auto import tqdm

from src.feature_extraction.eda import get_eda_features
from src.utils.io import load_config

data_segmented_left: list[tuple] = []
data_segmented_right: list[tuple] = []


_filename: str = basename(__file__).split(".")[0][4:]
basicConfig(filename=f"logs/{_filename}.log", level=INFO)
logger = getLogger(_filename)


from numpy import nan, ndarray
from pandas import Timedelta


def get_session_moment(start: Timestamp, end: Timestamp, info: Series) -> int | float:
    """
    Gets the session moment, i.e., if the person was awake, asleep or in between.
    The value will be 0 if the person was awake, 1 if the person was asleep and
    nan if the session given is in between.

    Parameters
    ----------
    start : Timestamp
        The start of the segment.
    end : Timestamp
        The end of the segment.
    info : Series
        The session info.

    Returns
    -------
    int | float
        The session moment.
    """
    # this method gives 0 if the person is awake and 1 if the person is
    if (start < info["bed_time"] and end < info["bed_time"]) or (
        start > info["wake_up_time"] and end > info["wake_up_time"]
    ):
        return 0
    elif (start < info["bed_time"] and end > info["bed_time"]) or (
        start < info["wake_up_time"] and end > info["wake_up_time"]
    ):
        return nan
    else:
        return 1


def get_segments(
    session_data: DataFrame,
    session_info: Series,
    starts: ndarray,
    ends: ndarray,
    session: str,
):
    """
    Gets the segments for a session.

    Parameters
    ----------
    session_data : DataFrame
        The session data.
    session_info : Series
        The session info.
    starts : ndarray
        The starts of the segments.
    ends : ndarray
        The ends of the segments.
    session : str
        The session.

    Returns
    -------
    list[tuple[ndarray, int | float]]
        The segments.
    """
    segments = [
        (
            session_data.loc[
                IndexSlice[session, start:end],
                "mixed-EDA",
            ].values,
            get_session_moment(start, end, session_info),
        )
        for start, end in zip(starts, ends)
    ]
    return segments


def get_starts_ends(
    session_data: DataFrame,
    segment_size_in_mins: int,
    segment_size_in_sampling_rate: int,
    data_sample_rate: int,
):
    """
    Gets the starts and ends of the segments.

    Parameters
    ----------
    session_data : DataFrame
        The session data.
    segment_size_in_mins : int
        The size of the segments in minutes.
    segment_size_in_sampling_rate : int
        The size of the segments in sampling rate units.
    data_sample_rate : int
        The sampling rate of the physiological data.

    Returns
    -------
    tuple[ndarray, ndarray]
        The starts and ends of the segments.
    """
    starts = session_data[::segment_size_in_sampling_rate].index.get_level_values(1)
    ends = (
        session_data[::segment_size_in_sampling_rate].index.get_level_values(1)
        + Timedelta(f"{segment_size_in_mins}min")
        - Timedelta(f"{1/data_sample_rate}s")
    )
    return starts, ends


def perform_segmentation(
    physiological_data: dict[str, dict[str, DataFrame]],
    experiment_info: DataFrame,
    experiment_info_as_dict: dict[str, DataFrame],
    segment_size_in_mins: int,
    segment_size_in_sampling_rate: int,
    data_sample_rate: int,
) -> tuple[list[tuple[ndarray, int | float]], list[tuple[ndarray, int | float]]]:
    """
    Performs segmentation of the physiological data.

    Parameters
    ----------
    physiological_data : dict[str, dict[str, DataFrame]]
        The physiological data to be segmented.
    experiment_info : DataFrame
        The experiment info.
    experiment_info_as_dict : dict[str, DataFrame]
        The experiment info as a dictionary.
    segment_size_in_mins : int
        The size of the segments in minutes.
    segment_size_in_sampling_rate : int
        The size of the segments in sampling rate units.
    data_sample_rate : int
        The sampling rate of the physiological data.

    Returns
    -------
    tuple[list[tuple[ndarray, int | float]], list[tuple[ndarray, int | float]]]
        The segmented data.
    """
    data_segmented_left: list[tuple] = []
    data_segmented_right: list[tuple] = []
    users = list(
        set(physiological_data["left"].keys()) & set(physiological_data["right"].keys())
    )

    for user in tqdm(users, desc="Splitting data. Users progress:", colours="blue"):
        # TODO: add a for loop for left and right: it would reduce the amount of
        # boilerplate code
        data_left = physiological_data["left"][user]
        data_right = physiological_data["right"][user]
        info = experiment_info_as_dict[user]
        sessions = list(
            set(data_left.index.get_level_values(0).unique())
            & set(data_right.index.get_level_values(0).unique())
        )
        morning_survey_sessions = (
            experiment_info.loc[IndexSlice[user, :], :]
            .index.get_level_values(1)
            .unique()
        )

        sessions_all = list(set(sessions) & set(morning_survey_sessions))

        for session in tqdm(
            sessions_all, desc=f"Splitting {user=}. Sessions progress:", colours="green"
        ):
            session_data_left: DataFrame = data_left.loc[IndexSlice[session, :], :]
            session_data_right: DataFrame = data_right.loc[IndexSlice[session, :], :]

            session_info = info.loc[session, :]

            starts_left, ends_left = get_starts_ends(
                session_data=session_data_left,
                segment_size_in_mins=segment_size_in_mins,
                segment_size_in_sampling_rate=segment_size_in_sampling_rate,
                data_sample_rate=data_sample_rate,
            )
            starts_right, ends_right = get_starts_ends(
                session_data=session_data_right,
                segment_size_in_mins=segment_size_in_mins,
                segment_size_in_sampling_rate=segment_size_in_sampling_rate,
                data_sample_rate=data_sample_rate,
            )

            segments_left = get_segments(
                session_data=session_data_left,
                session_info=session_info,
                starts=starts_left,
                ends=ends_left,
                session=session,
            )
            segments_right = get_segments(
                session_data=session_data_right,
                session_info=session_info,
                starts=starts_right,
                ends=ends_right,
                session=session,
            )

            data_segmented_left.append(segments_left)
            data_segmented_right.append(segments_right)

    return data_segmented_left, data_segmented_right


def get_values_labels(
    data_segmented: list[tuple[ndarray, int | float]]
) -> tuple[list[ndarray], list[ndarray]]:
    """
    Gets the values and labels from the segmented data.

    Parameters
    ----------
    data_segmented : list[tuple[ndarray, int | float]]
        The segmented data.

    Returns
    -------
    tuple[list[ndarray], list[int | float]]
        The values and labels.
    """
    values = [
        val[0] for segments in data_segmented for val in segments if not isnan(val[1])
    ]
    labels = [
        val[1] for segments in data_segmented for val in segments if not isnan(val[1])
    ]
    return values, labels


def main():
    path_to_config: str = f"src/run/pre_processing/config_{_filename}.yml"

    logger.info("Starting model training")
    configs = load_config(path=path_to_config)
    logger.debug("Configs loaded")

    segment_size_in_mins: float = float(configs["segment_size_in_mins"])
    data_sample_rate: int = configs["sample_rate"]
    plot_data_distribution: bool = configs["plot_data_distribution"]
    data_type: str = configs["data_type"]

    segment_size_in_secs = segment_size_in_mins * 60
    segment_size_in_sampling_rate: int = segment_size_in_secs * data_sample_rate

    physiological_data: dict = ...
    experiment_info: DataFrame = ...
    experiment_info_as_dict: dict = ...

    data_segmented_left, data_segmented_right = perform_segmentation(
        physiological_data=physiological_data,
        experiment_info=experiment_info,
        experiment_info_as_dict=experiment_info_as_dict,
        segment_size_in_mins=segment_size_in_mins,
        segment_size_in_sampling_rate=segment_size_in_sampling_rate,
        data_sample_rate=data_sample_rate,
    )
    values_left, labels_left = get_values_labels(data_segmented_left)
    values_right, labels_right = get_values_labels(data_segmented_right)

    if plot_data_distribution:
        # plot_distribution(labels_left, labels_right)
        # plot_distribution(labels_left, labels_right)
        raise NotImplementedError("Plotting data distribution is not implemented yet.")

    features_left = Parallel(n_jobs=-1)(
        delayed(get_eda_features)(value) for value in (values_left)
    )
    features_right = Parallel(n_jobs=-1)(
        delayed(get_eda_features)(value) for value in (values_right)
    )

    # TODO: save w/ pandas with column name
    features_left: ndarray = stack(features_left)
    features_right: ndarray = stack(features_right)
    labels_left: ndarray = stack(labels_left)
    labels_right: ndarray = stack(labels_right)

    savetxt(
        f"./data.nosync/features_{data_type}_left.csv", features_left, delimiter=","
    )
    savetxt(
        f"./data.nosync/features_{data_type}_right.csv", features_right, delimiter=","
    )
    savetxt(f"./data.nosync/labels_{data_type}_left.csv", labels_left, delimiter=",")
    savetxt(f"./data.nosync/labels_{data_type}_right.csv", labels_right, delimiter=",")
