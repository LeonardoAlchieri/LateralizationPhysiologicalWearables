from typing import Literal

from numpy import isnan, nan, ndarray
from pandas import DataFrame, IndexSlice, Series, Timedelta, Timestamp
from tqdm.auto import tqdm

from src.utils.misc import get_all_sessions


def get_session_moment(
    start: Timestamp, end: Timestamp, info: Series, mode: int = 1
) -> int | float:
    # TODO: check that this is working as expected
    # this method gives 0 if the person is awake and 1 if the person is
    if mode == 1:
        if (end < info["end_baseline_1"]) or (
            (start < info["end_baseline_1"] and end > info["end_baseline_1"])
        ):
            return 0
        elif (
            (start > info["end_baseline_1"] and end < info["start_cognitive_load"])
            or (start > info["end_cognitive_load"])
            or (
                start < info["start_cognitive_load"]
                and end > info["start_cognitive_load"]
            )
        ):
            return nan
        else:
            return 1
    elif mode == 2:
        # this method gives 0 if the person is awake and 1 if the person is
        if (start < info["actual_bed_time"] and end < info["actual_bed_time"]) or (
            start > info["wake_up_time"] and end > info["wake_up_time"]
        ):
            return 0
        elif (start < info["actual_bed_time"] and end > info["actual_bed_time"]) or (
            start < info["wake_up_time"] and end > info["wake_up_time"]
        ):
            return nan
        else:
            return 1


def organize_segmented_data(
    data_segmented_left: list[tuple], data_segmented_right: list[tuple], artefact: bool
) -> tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]:
    values_left = [
        val[0]
        for segments in data_segmented_left
        for val in segments
        if not isnan(val[1])
    ]
    values_right = [
        val[0]
        for segments in data_segmented_right
        for val in segments
        if not isnan(val[1])
    ]

    labels_left = [
        val[1]
        for segments in data_segmented_left
        for val in segments
        if not isnan(val[1])
    ]
    labels_right = [
        val[1]
        for segments in data_segmented_right
        for val in segments
        if not isnan(val[1])
    ]

    groups_left = [
        val[2]
        for segments in data_segmented_left
        for val in segments
        if not isnan(val[1])
    ]

    groups_right = [
        val[2]
        for segments in data_segmented_right
        for val in segments
        if not isnan(val[1])
    ]
    if not artefact:
        return (
            values_left,
            values_right,
            labels_left,
            labels_right,
            groups_left,
            groups_right,
        )
    else:
        artefacts_left = [
            val[3]
            for segments in data_segmented_left
            for val in segments
            if not isnan(val[1])
        ]

    artefacts_right = [
        val[3]
        for segments in data_segmented_right
        for val in segments
        if not isnan(val[1])
    ]
    return (
        values_left,
        values_right,
        labels_left,
        labels_right,
        groups_left,
        groups_right,
        artefacts_left,
        artefacts_right,
    )


def segment_info(
    session_data: DataFrame,
    session: str,
    start: Timestamp,
    end: Timestamp,
    session_info: Series,
    user: str,
    mode: int,
    artefact: bool,
):
    if not artefact:
        return (
            session_data.loc[
                IndexSlice[session, start:end],
                "mixed-EDA",
            ].values,
            get_session_moment(start, end, session_info, mode),
            user,
        )
    else:
        return (
            session_data.loc[
                IndexSlice[session, start:end],
                "mixed-EDA",
            ].values,
            get_session_moment(start, end, session_info, mode),
            user,
            session_data.loc[
                IndexSlice[session, start:end],
                "Artifact",
            ].values.max(),
        )


def segment(
    data: dict[Literal["left", "right"], dict[str, Series | DataFrame]],
    experiment_info_as_dict: dict[str, Series],
    segment_size_in_sampling_rate: int,
    segment_size_in_secs: int,
    data_sample_rate: int,
    # sessions_all: list[str] | None = ["experiment"],
    mode: int = 1,
    artefact: bool = False,
):
    data_segmented_left: list[tuple] = []
    data_segmented_right: list[tuple] = []
    users = list(set(data["left"].keys()) & set(data["right"].keys()))

    for user in tqdm(users, desc="User progress", colour="blue"):
        data_left = data["left"][user]
        data_right = data["right"][user]
        info = experiment_info_as_dict[user]

        # FIXME: using sessions like this seems stupid, but allows to reuse code
        # from the other experiment. I should however find a way around it
        sessions_all = get_all_sessions(
            user_data_left=data_left, user_data_right=data_right
        )

        for session in sessions_all:
            session_data_left: DataFrame = data_left.loc[IndexSlice[session, :], :]
            session_data_right: DataFrame = data_right.loc[IndexSlice[session, :], :]

            if sessions_all == ["experiment"]:
                session_info = info
            else:
                session_info = info.loc[IndexSlice[user, session]]

            starts_left = session_data_left[
                ::segment_size_in_sampling_rate
            ].index.get_level_values(1)
            starts_right = session_data_right[
                ::segment_size_in_sampling_rate
            ].index.get_level_values(1)

            ends_left = (
                session_data_left[
                    ::segment_size_in_sampling_rate
                ].index.get_level_values(1)
                + Timedelta(f"{segment_size_in_secs}s")
                - Timedelta(f"{1/data_sample_rate}s")
            )
            ends_right = (
                session_data_right[
                    ::segment_size_in_sampling_rate
                ].index.get_level_values(1)
                + Timedelta(f"{segment_size_in_secs}s")
                - Timedelta(f"{1/data_sample_rate}s")
            )

            segments_left = [
                segment_info(
                    session_info=session_info,
                    mode=mode,
                    artefact=artefact,
                    start=start,
                    end=end,
                    session=session,
                    user=user,
                    session_data=session_data_left,
                )
                for start, end in zip(starts_left, ends_left)
            ]
            segments_right = [
                segment_info(
                    session_info=session_info,
                    mode=mode,
                    artefact=artefact,
                    start=start,
                    end=end,
                    session=session,
                    user=user,
                    session_data=session_data_right,
                )
                for start, end in zip(starts_right, ends_right)
            ]
            data_segmented_left.append(segments_left)
            data_segmented_right.append(segments_right)

    return organize_segmented_data(data_segmented_left, data_segmented_right, artefact)
