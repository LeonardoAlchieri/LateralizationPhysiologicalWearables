from os.path import basename, dirname
from sys import path

from pandas import DataFrame, IndexSlice, read_csv, read_parquet, to_datetime

path.append(".")
from logging import DEBUG, INFO, basicConfig, getLogger

from src.utils.io import load_config, save_data

_filename: str = basename(__file__).split(".")[0][4:]
_path_to_file: str = dirname(__file__)

basicConfig(filename=f"logs/{_filename}.log", level=INFO)
logger = getLogger(_filename)


def move_event_to_columns(df: DataFrame) -> DataFrame:
    starts = {}
    ends = {}
    for event in df.index.get_level_values(1).unique():
        starts[f"start_{event}"] = df.loc[IndexSlice[:, event], "start"].values
        ends[f"end_{event}"] = df.loc[IndexSlice[:, event], "end"].values
    return DataFrame({**starts, **ends})


def main():
    path_to_config: str = f"{_path_to_file}/config_{_filename}.yml"

    logger.info("Starting model training")
    configs = load_config(path=path_to_config)
    logger.debug("Configs loaded")

    path_to_data: str = configs["path_to_data"]
    path_to_save_folder: str = configs["path_to_save_folder"]
    save_format: str = configs["save_format"]

    input_format: str | None = configs.get("input_format", None)
    if input_format is None:
        logger.warning(f"No input format specified. Trying to extract")
        input_format: str = path_to_data.split(".")[-1]
        logger.info(f"Extracted input format: {input_format}")
        
    if input_format == "csv":
        experiment_info = read_csv(path_to_data, index_col=[0, 1])
    elif input_format == "parquet":
        experiment_info = read_parquet(path_to_data)
    else:
        raise ValueError(f"Input format {input_format} not supported. Supported formats are: csv, parquet")
        
    indexes_to_drop = [
        idx
        for idx in experiment_info.index
        if "baseline" not in idx[1] and "cognitive_load" not in idx[1]
    ]
    experiment_info = experiment_info.drop(indexes_to_drop, inplace=False)
    experiment_info = experiment_info.groupby(axis=0, level=0, group_keys=True).apply(
        move_event_to_columns
    )
    experiment_info.index = experiment_info.index.droplevel(1)
    experiment_info = experiment_info.applymap(to_datetime)
    experiment_info = experiment_info.applymap(lambda x: x.tz_localize("Europe/Rome"))
    experiment_info = experiment_info[["start_baseline_1", "end_baseline_5"]]
    experiment_info = experiment_info.rename({"start_baseline_1": "start", "end_baseline_5": "end"}, axis=1)

    save_data(
        data_to_save=experiment_info,
        filepath=f"{path_to_save_folder}/experiment_time",
        save_format=save_format,
    )


if __name__ == "__main__":
    main()
