from glob import glob
from os import remove as remove_file
from os.path import join as join_paths
from pathlib import Path

# from joblib import Parallel, delayed
from random import choice as choose_randomly
from sys import path
from time import time
from typing import Any
from warnings import warn

from eda_artefact_detection.detection import compute_eda_artifacts
from numpy import ndarray, stack
from pandas import DataFrame, IndexSlice, Series, concat, read_csv
from tqdm.auto import tqdm

path.append(".")
from collections import defaultdict
from logging import DEBUG, INFO, basicConfig, getLogger

from src.utils import (
    blockPrinting,
    make_timestamp_idx,
    parallel_iteration,
    remove_empty_sessions,
    segment_over_experiment_time,
)
from src.utils.eda import decomposition
from src.utils.filters import butter_lowpass_filter_filtfilt
from src.utils.io import load_and_prepare_data, load_config, load_processed_data
from src.utils.plots import make_lineplot
from src.utils.pre_processing import (
    concate_session_data,
    rescaling,
    get_rescaling_technique,
)

basicConfig(filename="logs/run_eda_filtering.log", level=DEBUG)

logger = getLogger("main")


@parallel_iteration
@blockPrinting
def gashis_artefact_detection(
    data: DataFrame | Series,
    # n_jobs: int = 1,
    window_size: int = 4,
    **kwargs,
):
    return compute_eda_artifacts(
        data=data,
        show_database=True,
        convert_dataframe=True,
        output_path=None,
        window_size=window_size,
        return_vals=True,
    )[1].set_index("Time", inplace=False)


@parallel_iteration
def acc_artefact_detection(
    acc_magitude_data: dict[str, dict[str, dict[str, Series | DataFrame]]],
    session_data: DataFrame | Series,
    n_jobs: int = 1,
    acc_threshold: float = 0.9,
    **kwargs,
) -> DataFrame:
    side_name: str = kwargs["side_name"]
    user_name: str = kwargs["user_name"]
    if user_name in acc_magitude_data[side_name].keys():
        current_acc_data: DataFrame = acc_magitude_data[side_name][user_name]
        bool_mask = current_acc_data > acc_threshold
        if isinstance(session_data, DataFrame):
            session_data["Artifact"] = bool_mask.astype(int)
        elif isinstance(session_data, Series):
            session_data = session_data.to_frame()
            session_data.columns = ["EDA"]
            # FIXME: the acc data has a different sampling rate, and as such I cannot match the indexes
            # I need to undersample to get the same indices!
            session_data["Artifact"] = bool_mask.astype(int)
        else:
            raise TypeError(f'Unknown data type "{type(session_data)}"')
    else:
        pass
    return session_data


def main():
    path_to_config: str = "src/run/pre_processing/config_eda_filtering.yml"

    logger.info("Starting model training")
    configs: dict[str, Any] = load_config(path=path_to_config)
    logger.debug("Configs loaded")

    path_to_main_folder: str = configs["path_to_main_folder"]
    path_to_save_folder: str = configs["path_to_save_folder"]
    path_to_experiment_time: str | None = configs.get("path_to_experiment_time", None)
    rescaling_method_name: str = configs["rescaling_method"]
    cutoff_frequency: float = configs["cutoff_frequency"]
    butterworth_order: int = configs["butterworth_order"]
    n_jobs: int = configs["n_jobs"]
    plots: bool = configs["plots"]
    clean_plots: bool = configs["clean_plots"]
    mode: int = configs["mode"]
    device: str = configs["device"]
    concat_sessions: bool = configs["concat_sessions"]
    subset_data: bool = configs["subset_data"]
    artefact_detection: int = configs["artefact_detection"]
    artefact_window_size: int = configs.get("artefact_window_size", None)

    if clean_plots:
        files_to_remove = glob("./visualizations/EDA/*.pdf")
        for f in files_to_remove:
            remove_file(f)
        del files_to_remove

    experiment_time: DataFrame | None
    if path_to_experiment_time is None:
        logger.warning(
            f'No path to experiment time provided. Not applying filter "segment_over_experiment_time".'
        )
        experiment_time = None
    else:
        # TODO: add check for other file formats
        experiment_time = read_csv(path_to_experiment_time, index_col=0)

    eda_data = load_and_prepare_data(
        path_to_main_folder=path_to_main_folder,
        side=None,
        data_type="EDA",
        mode=mode,
        device=device,
    )

    if subset_data:
        eda_data_new = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        warn("Subsetting data to 1000 samples per session.")
        i = 0
        for side in eda_data.keys():
            for user in eda_data[side].keys():
                for session in eda_data[side][user].keys():
                    eda_data_new[side][user][session] = eda_data[side][user][session]
                    i += 1
                    if i == 10:
                        break
                if i == 10:
                    break
            if i == 10:
                break
        eda_data = eda_data_new
        del eda_data_new
        # [:5000]

    if artefact_detection == 1:
        print(f"Using artefact detection method 1.")
        if artefact_window_size is None:
            raise ValueError(
                f"Artefact window size must be provided when using artefact detection method 1."
            )
        # FIXME: not working anymore
        eda_data = gashis_artefact_detection(
            data=eda_data, window_size=artefact_window_size, n_jobs=n_jobs
        )
    else:
        eda_data = {
            side: {
                user: {
                    session: make_timestamp_idx(dataframe=session_data, data_name="EDA")
                    for session, session_data in eda_data[side][user].items()
                }
                for user in eda_data[side].keys()
            }
            for side in eda_data.keys()
        }

        if artefact_detection == 2:
            acc_data_path: str | None = configs.get("acc_data_path", None)
            if acc_data_path is None:
                raise ValueError(
                    f"When using artefact detection 2, acc_data_path must be provided. Received {acc_data_path}"
                )
            acc_threshold: float = configs.get("acc_threshold", 0.9)
            acc_data = load_processed_data(path=acc_data_path, file_format="parquet")

            eda_data = acc_artefact_detection(
                data=eda_data,
                n_jobs=1,
                acc_magitude_data=acc_data,
                acc_threshold=acc_threshold,
            )
        elif artefact_detection == 0:
            logger.info("No artefact implementation")
        else:
            raise ValueError(
                f"Artefact detection method {artefact_detection} not recognized. Please choose between 0, 1 and 2."
            )

    # NOTE: segmentation over the experiment time has to happen after the
    # timestamp is made as index, since it is required for the segmentation
    if experiment_time is not None:
        eda_data = segment_over_experiment_time(eda_data, experiment_time)
    eda_data = remove_empty_sessions(eda_data)

    # NOTE: the data here is order this way: {side: {user: session: {Series}}},
    # ir {side: {user: Series}}, depending on the chosen mode.
    # Each pandas Series contains also the `attr` field  the
    # metadata relative to the specific user <-- pretty sure I did
    # not implement this at the end

    logger.info("Data loaded correctly.")
    logger.info(f"Number of sides: {len(eda_data.keys())}")
    logger.info(f"Number of users for right side: {len(eda_data['right'].keys())}")
    logger.info(f"Number of users for left side: {len(eda_data['left'].keys())}")

    if plots:
        random_side: str = choose_randomly(list(eda_data.keys()))
        random_user: str = choose_randomly(list(eda_data[random_side].keys()))
        random_session: str = choose_randomly(
            list(eda_data[random_side][random_user].keys())
        )
        logger.info(f"Making plots for side {random_side} and user {random_user}")
        make_lineplot(
            data=eda_data[random_side][random_user][random_session],
            which="EDA",
            savename=f"eda_{random_side}_{random_user}_{random_session}",
            title="Example EDA",
        )

    eda_data_filtered: defaultdict[str, dict[str, dict[str, Series | DataFrame]]] = {
        side: {
            user: {
                session_name: DataFrame(
                    butter_lowpass_filter_filtfilt(
                        data=session_data,
                        cutoff=cutoff_frequency,
                        fs=session_data.attrs["sampling frequency"],
                        order=butterworth_order,
                    ),
                    index=session_data.index,
                    # NOTE: if we run the artefact detection, it is going to be a dataframe
                    columns=session_data.columns
                    if isinstance(session_data, DataFrame)
                    else None,
                )
                for session_name, session_data in user_edat_data.items()
            }
            for user, user_edat_data in tqdm(
                eda_data[side].items(),
                desc=f'Filtering EDA data for side "{side}"',
                colour="green",
            )
        }
        for side in eda_data.keys()
    }

    if plots:
        make_lineplot(
            data=eda_data_filtered[random_side][random_user][random_session],
            which="EDA",
            savename=f"eda_filtered_{random_side}_{random_user}_{random_session}",
            title="Example EDA after filter",
        )

    start = time()

    eda_data_phasic: defaultdict[str, list[dict[str, ndarray]]] = {
        side: {
            user: {
                session: DataFrame(
                    decomposition(
                        session_data.values,
                        eda_data[side][user][session].attrs["sampling frequency"],
                    )["phasic component"]
                    if isinstance(session_data, Series)
                    or (
                        isinstance(session_data, DataFrame)
                        and len(session_data.columns) == 1
                    )
                    else (
                        stack(
                            [
                                decomposition(
                                    session_data.values,
                                    eda_data[side][user][session].attrs[
                                        "sampling frequency"
                                    ],
                                )["phasic component"],
                                session_data[:, 1]
                                if isinstance(session_data, ndarray)
                                else session_data.iloc[:, 1],
                            ],
                            axis=1,
                        )
                    ),
                    index=session_data.index,
                    columns=(
                        session_data.columns
                        if isinstance(session_data, DataFrame)
                        else None
                    ),
                )
                for session, session_data in tqdm(
                    user_edat_data.items(), desc="Session progress", colour="blue"
                )
            }
            for user, user_edat_data in tqdm(
                eda_data_filtered[side].items(),
                desc="EDA decomposition progress (user)",
                colour="green",
            )
        }
        for side in eda_data_filtered.keys()
    }

    print("Total phasic component calculation: %.2f s" % (time() - start))
    if plots:
        make_lineplot(
            data=eda_data_phasic[random_side][random_user][random_session],
            which="EDA",
            savename=f"eda_phasic_{random_side}_{random_user}_{random_session}",
            title="Example EDA phasic component",
        )

    eda_data_standardized = rescaling(
        data=eda_data_filtered,
        rescaling_method=get_rescaling_technique(rescaling_method_name),
    )
    eda_data_standardized_phasic = rescaling(
        data=eda_data_phasic,
        rescaling_method=get_rescaling_technique(rescaling_method_name),
    )

    if plots:
        make_lineplot(
            data=eda_data_standardized[random_side][random_user][random_session],
            which="EDA",
            savename=f"eda_standardized_{random_side}_{random_user}_{random_session}",
            title="Example EDA filtered & standardized",
        )
        make_lineplot(
            data=eda_data_standardized_phasic[random_side][random_user][random_session],
            which="EDA",
            savename=f"eda_standardized_{random_side}_{random_user}_{random_session}",
            title="Example EDA phasic standardized",
        )

    if concat_sessions:
        eda_data_standardized_phasic = concate_session_data(
            eda_data_standardized_phasic
        )

        eda_data_standardized = concate_session_data(eda_data_standardized)

        # TODO: the code should be able to handle even when there is no session concatenation
        for side in eda_data_standardized.keys():
            if side in eda_data_standardized_phasic.keys():
                for user in tqdm(
                    eda_data_standardized[side].keys(),
                    desc=f'Saving EDA data for side "{side}"',
                ):
                    if user in eda_data_phasic[side].keys():
                        user_data_standardized: Series = eda_data_standardized[side][
                            user
                        ]
                        user_data_phasic: Series = eda_data_standardized_phasic[side][
                            user
                        ]
                        df_to_save: DataFrame = concat(
                            [user_data_standardized, user_data_phasic], axis=1
                        )
                        if "Artifact" in df_to_save.columns:
                            aux_res = df_to_save[["Artifact"]].loc[
                                :, ~df_to_save[["Artifact"]].columns.duplicated()
                            ]
                            df_to_save = df_to_save.drop(
                                columns=["Artifact"], inplace=False
                            )
                            df_to_save["Artifact"] = aux_res

                        if len(df_to_save.columns) == 2:
                            df_to_save.columns = ["mixed-EDA", "phasic-EDA"]
                        elif len(df_to_save.columns) == 3:
                            df_to_save.columns = ["mixed-EDA", "phasic-EDA", "Artifact"]
                        else:
                            raise RuntimeError(
                                f"Unexpected number of columns: {len(df_to_save.columns)}"
                            )
                        # NOTE: I don't need the attributes, since I put all of the sessions together
                        # df_to_save.attrs['sampling rate'] = eda_data[side][user].attrs
                        # if plots and (side == random_side) and (user == random_user):
                        #     df_to_save["EDA original"] = standardize(
                        #         eda_data[random_side][random_user]
                        #     )
                        #     make_lineplot(
                        #         data=df_to_save,
                        #         which="EDA",
                        #         savename=f"eda_filteredXphasic_{random_side}_{random_user}",
                        #         title="Example EDA filtered, phasic & original (all standardized)",
                        #     )
                        path_to_save: str = f"{path_to_save_folder}/{side}/EDA/"
                        filename: str = f"{user}.parquet"

                        Path(path_to_save).mkdir(parents=True, exist_ok=True)

                        logger.info(
                            f'Saving EDA data for user "{user}" in "{path_to_save} to {filename}"'
                        )
                        df_to_save.to_parquet(
                            join_paths(path_to_save, filename),
                        )
                    else:
                        raise RuntimeError(
                            f"User {user} not found in eda_data_phasic for side {side}: {eda_data_phasic[side].keys()}"
                        )
            else:
                raise RuntimeError(
                    f"Side {side} not found in eda_data_phasic keys: {eda_data_phasic.keys()}"
                )


if __name__ == "__main__":
    main()
