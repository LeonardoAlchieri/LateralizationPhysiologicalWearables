# This script just takes the different csv files for the experiment info and puts them together
from glob import glob
from typing import Any
from logging import DEBUG, INFO, basicConfig, getLogger
from os.path import basename
from os.path import join as join_paths
from sys import path
from warnings import warn

from pandas import (
    DataFrame,
    Series,
    Timedelta,
    concat,
    isna,
    read_csv,
    to_datetime,
    Timestamp,
)

path.append(".")
from src.utils.io import load_config, save_data

_filename: str = basename(__file__).split(".")[0][4:]

basicConfig(filename=f"logs/{_filename}.log", level=INFO)
logger = getLogger(_filename)


# calculate_actual_bed_time()
# This function takes a row of a DataFrame and calculates the time the
# user actually went to bed. The actual bed time is the time the user
# said they went to bed plus the latency. The latency is the amount of
# time the user took to actually fall asleep.


def check_latency_consistency(latency: Any) -> int:
    if isinstance(latency, str):
        try:
            latency = float(latency)
        except ValueError:
            warn(f"Latency {latency} is not a number. Trying to split on -")
            try:
                latency = float(latency.split("-")[-1])
            except Exception:
                warn(f"Could not convert. Setting to 0")
                latency = 0
    # TODO: implement other checks
    return latency


def calculate_actual_bed_time(df_row: Series) -> Series:
    """this method computes the actual bed time, taken from the reported bed
    time and the bed time latency.

    Parameters
    ----------
    df_row : Series
        series for a single usersurvey

    Returns
    -------
    Series
        series for a single usersurvey with the actual bed time
    """
    if isna(df_row["bed_time"]):
        warn("Row is NaN. Returning NaN")
        return df_row

    bed_time: Timestamp = df_row["bed_time"]
    if isinstance(bed_time, str):
        bed_time: Timestamp = to_datetime(bed_time)
        current_date: Timestamp = to_datetime(df_row.name[-1]).date()
        bed_time.replace(
            year=current_date.year, month=current_date.month, day=current_date.day
        )
    correct_bed_time: Timestamp = bed_time + Timedelta(
        minutes=check_latency_consistency(df_row["latency"])
    )
    if correct_bed_time.time() < to_datetime("12:00:00").time():
        pass
    else:
        correct_bed_time += Timedelta(days=-1)
    return correct_bed_time


# This function converts the bed and wake-up times from strings to timestamps.
# The bed and wake-up times are in the format "hh:mm:ss".
# The date is in the format "YYYY-MM-DD".
# The function returns a dataframe row with the bed and wake-up times converted to timestamps.
def make_times_into_timestamps(df_row: Series) -> Series:
    """Converts the time columns in a row of a dataframe
    into timestamps.

    Parameters
    ----------
    df_row : Series
        A row of a dataframe, containing one row of data

    Returns
    -------
    Series
        A row of a dataframe, containing one row of data, with
        the time columns converted into timestamps
    """
    # Extract the date from the row.
    date: str = df_row["Date"]

    # Extract the bed and wake-up times from the row.
    bed_time: str = df_row["bed_time"]
    wake_up_time: str = df_row["wake_up_time"]

    # If either time is missing, return the row unchanged.
    if isna(bed_time) or isna(wake_up_time):
        return df_row

    # Convert the bed and wake-up times to timestamps.
    df_row["bed_time"] = to_datetime(f"{date} {bed_time}")
    df_row["wake_up_time"] = to_datetime(f"{date} {wake_up_time}")
    return df_row


def main():
    path_to_config: str = (
        f"src/run/dataset_specific/sleep_lateralization/config_{_filename}.yml"
    )

    logger.info("Starting model training")
    configs = load_config(path=path_to_config)
    logger.debug("Configs loaded")

    path_to_data: str = configs["path_to_data"]
    path_to_save_folder: str = configs["path_to_save_folder"]
    save_format: str = configs["save_format"]

    all_experimentinfo_paths: list[str] = glob(
        join_paths(path_to_data, "*/*/morning_survey.csv")
    )
    logger.debug(f"All paths identified: {all_experimentinfo_paths}")
    logger.info("Loading and joining experiment info")
    all_experimentinfo_joined: DataFrame = concat(
        [read_csv(path, index_col=0) for path in all_experimentinfo_paths]
    )

    all_experimentinfo_joined = all_experimentinfo_joined.rename(
        columns={"bed_time ": "bed_time"}, inplace=False
    )
    all_experimentinfo_joined = all_experimentinfo_joined.reset_index(drop=False)

    # TODO: ask Nouran if timestamp is UTC or local
    all_experimentinfo_joined = all_experimentinfo_joined.apply(
        make_times_into_timestamps, axis=1, result_type="broadcast"
    )
    all_experimentinfo_joined = all_experimentinfo_joined.set_index(
        ["participant_id", "Date"]
    )

    all_experimentinfo_joined["actual_bed_time"] = all_experimentinfo_joined.apply(
        calculate_actual_bed_time, axis=1
    )

    all_experimentinfo_joined.dropna(axis=0, inplace=True, how="any")
    # TODO: add Oura Ring bedtime
    save_data(
        data_to_save=all_experimentinfo_joined,
        filepath=f"{path_to_save_folder}/all_experimento_info",
        save_format=save_format,
    )


if __name__ == "__main__":
    main()
