from glob import glob
from os import remove as remove_file
from os.path import join as join_paths
from pathlib import Path

# from joblib import Parallel, delayed
from random import choice as choose_randomly
from sys import path
from time import time
from warnings import warn

from pandas import DataFrame, Series, read_csv
from tqdm import tqdm

path.append(".")
from collections import defaultdict
from logging import DEBUG, INFO, basicConfig, getLogger

from src.utils import make_timestamp_idx, segment_over_experiment_time
from src.utils.pre_processing import concate_session_data, rescaling
from src.utils.pre_processing import standardize
from src.utils.filters import butter_lowpass_filter_lfilter
from src.utils.io import load_and_prepare_data, load_config
from src.utils.plots import make_lineplot

basicConfig(filename="logs/run_bvp_filtering.log", level=DEBUG)

logger = getLogger("main")


def main():
    path_to_config: str = "src/run/pre_processing/config_bvp_filtering.yml"

    logger.info("Starting model training")
    configs = load_config(path=path_to_config)
    logger.debug("Configs loaded")

    path_to_main_folder: str = configs["path_to_main_folder"]
    path_to_save_folder: str = configs["path_to_save_folder"]
    cutoff_frequency: float = configs["cutoff_frequency"]
    butterworth_order: int = configs["butterworth_order"]
    plots: bool = configs["plots"]
    clean_plots: bool = configs["clean_plots"]
    mode: int = configs["mode"]
    device: str = configs["device"]
    concat_sessions: bool = configs["concat_sessions"]
    subset_data: bool = configs["subset_data"]
    path_to_experiment_time: str = configs.get("path_to_experiment_time", None)

    if clean_plots:
        files_to_remove = glob("./visualizations/BVP/*.pdf")
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

    bvp_data = load_and_prepare_data(
        path_to_main_folder=path_to_main_folder,
        side=None,
        data_type="BVP",
        mode=mode,
        device=device,
    )

    if subset_data:
        warn("Subsetting data to 1000 samples per session.")
        for side in bvp_data.keys():
            for user in bvp_data[side].keys():
                for session in bvp_data[side][user].keys():
                    bvp_data[side][user][session] = bvp_data[side][user][session][:1000]

    bvp_data = {
        side: {
            user: {
                session: make_timestamp_idx(
                    dataframe=session_data,
                    data_name="BVP",
                )
                for session, session_data in bvp_data[side][user].items()
            }
            for user in bvp_data[side].keys()
        }
        for side in bvp_data.keys()
    }
    # NOTE: segmentation over the experiment time has to happen after the
    # timestamp is made as index, since it is required for the segmentation
    if experiment_time is not None:
        bvp_data = segment_over_experiment_time(bvp_data, experiment_time)
    # NOTE: the data here is order this way: {side: {user: session: {Series}}},
    # ir {side: {user: Series}}, depending on the chosen mode.
    # Each pandas Series contains also the `attr` field with the
    # metadata relative to the specific user <-- pretty sure I did
    # not implement this at the end

    logger.info("Data loaded correctly.")
    logger.info(f"Number of sides: {len(bvp_data.keys())}")
    logger.info(f"Number of users for right side: {len(bvp_data['right'].keys())}")
    logger.info(f"Number of users for left side: {len(bvp_data['left'].keys())}")

    if plots:
        random_side: str = choose_randomly(list(bvp_data.keys()))
        random_user: str = choose_randomly(list(bvp_data[random_side].keys()))
        random_session: str = choose_randomly(
            list(bvp_data[random_side][random_user].keys())
        )
        logger.info(f"Making plots for side {random_side} and user {random_user}")
        make_lineplot(
            data=bvp_data[random_side][random_user][random_session],
            which="BVP",
            savename=f"bvp_{random_side}_{random_user}_{random_session}",
            title="Example BVP",
        )

    bvp_data_filtered: defaultdict[str, dict[str, dict[str, Series]]] = {
        side: {
            user: {
                session_name: Series(
                    butter_lowpass_filter_lfilter(
                        data=session_data,
                        cutoff=cutoff_frequency,
                        fs=session_data.attrs["sampling frequency"],
                        order=butterworth_order,
                    ),
                    index=session_data.index,
                )
                for session_name, session_data in user_bvpt_data.items()
            }
            for user, user_bvpt_data in tqdm(
                bvp_data[side].items(),
                desc=f'Filtering BVP data for side "{side}"',
                colour="green",
            )
        }
        for side in bvp_data.keys()
    }

    if plots:
        make_lineplot(
            data=bvp_data_filtered[random_side][random_user][random_session],
            which="BVP",
            savename=f"bvp_filtered_{random_side}_{random_user}_{random_session}",
            title="Example BVP after filter",
        )

    bvp_data_standardized = rescaling(
        data=bvp_data_filtered, rescaling_method=standardize
    )

    if plots:
        make_lineplot(
            data=bvp_data_standardized[random_side][random_user][random_session],
            which="BVP",
            savename=f"bvp_standardized_{random_side}_{random_user}_{random_session}",
            title="Example BVP phasic standardized",
        )

    if concat_sessions:
        start = time()
        # FIXME: wrong session name recorded
        bvp_data_standardized = concate_session_data(bvp_data_standardized)

        logger.info("Concatenating session data took %.2fs" % (time() - start))

        logger.info("Finished concatenating session data")

        # TODO: the code should be able to handle even when there is no session concatenation
        for side in bvp_data_standardized.keys():
            for user in tqdm(
                bvp_data_standardized[side].keys(),
                desc=f'Saving BVP data for side "{side}"',
            ):
                user_data_standardized: Series = bvp_data_standardized[side][user]
                df_to_save: DataFrame = DataFrame(
                    user_data_standardized, columns=["BVP"]
                )

                path_to_save: str = f"{path_to_save_folder}/{side}/BVP/"
                filename: str = f"{user}.parquet"

                Path(path_to_save).mkdir(parents=True, exist_ok=True)

                df_to_save.to_parquet(
                    join_paths(path_to_save, filename),
                )
    else:
        NotImplementedError(
            "The code does not handle the case when there is no session concatenation yet."
        )


if __name__ == "__main__":
    main()
