from glob import glob
from os import remove as remove_file
from os.path import join as join_paths
from pathlib import Path

# from joblib import Parallel, delayed
from random import choice as choose_randomly
from sys import path
from time import time
from warnings import warn

from joblib import Parallel, delayed
from pandas import DataFrame, Series, read_csv
from tqdm.auto import tqdm

path.append(".")
from collections import defaultdict
from logging import DEBUG, INFO, basicConfig, getLogger

from src.utils import make_timestamp_idx, segment_over_experiment_time
from src.utils.filters import moving_avg_acc
from src.utils.io import load_and_prepare_data, load_config
from src.utils.plots import make_lineplot
from src.utils.pre_processing import concate_session_data, rescaling, standardize

basicConfig(filename="logs/run_acc_filtering.log", level=DEBUG)

logger = getLogger("main")


def apply_moving_avg_filtering(
    acc_data: defaultdict[str, dict[str, dict[str, Series]]],
    window_size: int,
    n_jobs: int = 1,
) -> dict[str, dict[str, dict[str, Series]]]:
    if n_jobs == 1:
        results = {
            side: {
                user: {
                    session_name: DataFrame(
                        moving_avg_acc(
                            data=session_data,
                            window_size=window_size,
                        ),
                        index=session_data.index,
                        columns=session_data.columns if isinstance(session_data, DataFrame) else None,
                    )
                    for session_name, session_data in user_acct_data.items()
                }
                for user, user_acct_data in tqdm(
                    acc_data[side].items(),
                    desc=f'Filtering ACC data for side "{side}"',
                    colour="green",
                )
            }
            for side in acc_data.keys()
        }
    elif n_jobs > 1 or n_jobs == -1:

        def support_moving_avg_acc(
            session_data: DataFrame, session_name: str
        ) -> tuple[str, Series]:
            return (
                session_name,
                DataFrame(
                    moving_avg_acc(
                        data=session_data,
                        window_size=window_size,
                    ),
                    index=session_data.index,
                    columns=['ACC'],
                ),
            )

        results = {
            side: {
                user: Parallel(n_jobs=n_jobs)(
                    delayed(support_moving_avg_acc)(session_data, session_name)
                    for session_name, session_data in user_acct_data.items()
                )
                for user, user_acct_data in tqdm(
                    acc_data[side].items(),
                    desc=f'Filtering ACC data for side "{side}"',
                    colour="green",
                )
            }
            for side in acc_data.keys()
        }

        results = {
            side: {
                user: {
                    session_name: session_data
                    for (session_name, session_data) in user_acct_data
                }
                for user, user_acct_data in results[side].items()
            }
            for side in results.keys()
        }
    else:
        raise ValueError(f'Invalid value for "n_jobs": got {n_jobs}')

    return results


def main():
    path_to_config: str = "src/run/pre_processing/config_acc_filtering.yml"

    logger.info("Starting model training")
    configs = load_config(path=path_to_config)
    logger.debug("Configs loaded")

    path_to_main_folder: str = configs["path_to_main_folder"]
    path_to_save_folder: str = configs["path_to_save_folder"]
    window_size: int = configs["window_size"]
    plots: bool = configs["plots"]
    clean_plots: bool = configs["clean_plots"]
    mode: int = configs["mode"]
    device: str = configs["device"]
    concat_sessions: bool = configs["concat_sessions"]
    subset_data: bool = configs["subset_data"]
    n_jobs: int = configs.get("n_jobs", 1)
    path_to_experiment_time: str | None = configs.get("path_to_experiment_time", None)
    savetype: str = configs.get("savetype", "csv")

    if clean_plots:
        files_to_remove = glob("./visualizations/ACC/*.pdf")
        for f in files_to_remove:
            remove_file(f)
        del files_to_remove

    if path_to_experiment_time is not None:
        experiment_time = read_csv(path_to_experiment_time, index_col=0)
    else:
        experiment_time = None

    acc_data = load_and_prepare_data(
        path_to_main_folder=path_to_main_folder,
        side=None,
        data_type="ACC",
        mode=mode,
        device=device,
    )

    if subset_data:
        warn("Subsetting data to 1000 samples per session.")
        for side in acc_data.keys():
            for user in acc_data[side].keys():
                for session in acc_data[side][user].keys():
                    acc_data[side][user][session] = acc_data[side][user][session][:1000]

    acc_data = {
        side: {
            user: {
                session: make_timestamp_idx(
                    dataframe=session_data,
                    data_name="ACC",
                )
                for session, session_data in acc_data[side][user].items()
            }
            for user in acc_data[side].keys()
        }
        for side in acc_data.keys()
    }
    # NOTE: segmentation over the experiment time has to happen after the
    # timestamp is made as index, since it is required for the segmentation
    if experiment_time is not None:
        acc_data = segment_over_experiment_time(acc_data, experiment_time)
    else:
        logger.info(f"No experiment time file provided, skipping segmentation.")
    # NOTE: the data here is order this way: {side: {user: session: {Series}}},
    # ir {side: {user: Series}}, depending on the chosen mode.
    # Each pandas Series contains also the `attr` field with the
    # metadata relative to the specific user <-- pretty sure I did
    # not implement this at the end

    logger.info("Data loaded correctly.")
    logger.info(f"Number of sides: {len(acc_data.keys())}")
    logger.info(f"Number of users for right side: {len(acc_data['right'].keys())}")
    logger.info(f"Number of users for left side: {len(acc_data['left'].keys())}")

    if plots:
        random_side: str = choose_randomly(list(acc_data.keys()))
        random_user: str = choose_randomly(list(acc_data[random_side].keys()))
        random_session: str = choose_randomly(
            list(acc_data[random_side][random_user].keys())
        )
        logger.info(f"Making plots for side {random_side} and user {random_user}")
        make_lineplot(
            data=acc_data[random_side][random_user][random_session],
            which="ACC",
            savename=f"acc_{random_side}_{random_user}_{random_session}",
            title="Example ACC",
        )

    acc_data_filtered = apply_moving_avg_filtering(
        acc_data=acc_data, window_size=window_size, n_jobs=n_jobs
    )

    if plots:
        make_lineplot(
            data=acc_data_filtered[random_side][random_user][random_session],
            which="ACC",
            savename=f"acc_filtered_{random_side}_{random_user}_{random_session}",
            title="Example ACC after filter",
        )

    acc_data_standardized = rescaling(
        data=acc_data_filtered, rescaling_method=standardize, n_jobs=n_jobs
    )

    if plots:
        make_lineplot(
            data=acc_data_standardized[random_side][random_user][random_session],
            which="ACC",
            savename=f"acc_standardized_{random_side}_{random_user}_{random_session}",
            title="Example ACC phasic standardized",
        )

    if concat_sessions:
        start = time()
        acc_data_standardized = concate_session_data(acc_data_standardized)

        logger.info("Concatenating session data took %.2fs" % (time() - start))

        logger.info("Finished concatenating session data")

        # TODO: the code should be able to handle even when there is no session concatenation
        for side in acc_data_standardized.keys():
            for user in tqdm(
                acc_data_standardized[side].keys(),
                desc=f'Saving ACC data for side "{side}"',
            ):
                user_data_standardized: Series = acc_data_standardized[side][user]
                df_to_save: DataFrame = DataFrame(
                    user_data_standardized, columns=["ACC"]
                )

                path_to_save: str = f"{path_to_save_folder}/{side}/ACC/"
                if savetype == "parquet":
                    filename: str = f"{user}.parquet"

                    Path(path_to_save).mkdir(parents=True, exist_ok=True)

                    df_to_save.to_parquet(
                        join_paths(path_to_save, filename),
                    )
                elif savetype == "csv":
                    filename: str = f"{user}.csv"

                    Path(path_to_save).mkdir(parents=True, exist_ok=True)

                    df_to_save.to_csv(
                        join_paths(path_to_save, filename),
                    )
                else:
                    raise ValueError(f'Unknown savetype "{savetype}"')
    else:
        NotImplementedError(
            "The code does not handle the case when there is no session concatenation yet."
        )


if __name__ == "__main__":
    main()
