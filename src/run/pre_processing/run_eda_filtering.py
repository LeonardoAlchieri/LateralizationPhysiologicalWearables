from glob import glob
from os import remove as remove_file
from os.path import join as join_paths
from pathlib import Path

# from joblib import Parallel, delayed
from random import choice as choose_randomly
from sys import path
from time import time
from warnings import warn

from numpy import ndarray
from pandas import DataFrame, Series, concat
from tqdm import tqdm

path.append(".")
from collections import defaultdict
from logging import DEBUG, INFO, basicConfig, getLogger

from src.utils import make_timestamp_idx, prepare_data_for_concatenation
from src.utils.eda import decomposition, standardize
from src.utils.filters import butter_lowpass_filter_filtfilt
from src.utils.io import load_and_prepare_data, load_config
from src.utils.plots import make_lineplot

basicConfig(filename="run_eda_filtering.log", level=DEBUG)

logger = getLogger("main")


def main():
    path_to_config: str = "src/run/config_eda_filtering.yml"

    logger.info("Starting model training")
    configs = load_config(path=path_to_config)
    logger.debug("Configs loaded")

    path_to_main_folder: str = configs["path_to_main_folder"]
    path_to_save_folder: str = configs["path_to_save_folder"]
    cutoff_frequency: float = configs["cutoff_frequency"]
    butterworth_order: int = configs["butterworth_order"]
    n_jobs: int = configs["n_jobs"]
    plots: bool = configs["plots"]
    clean_plots: bool = configs["clean_plots"]
    mode: int = configs["mode"]
    device: str = configs["device"]
    concat_sessions: bool = configs["concat_sessions"]
    subset_data: bool = configs["subset_data"]

    if clean_plots:
        files_to_remove = glob("./visualizations/EDA/*.pdf")
        for f in files_to_remove:
            remove_file(f)
        del files_to_remove

    eda_data = load_and_prepare_data(
        path_to_main_folder=path_to_main_folder,
        side=None,
        data_type="EDA",
        mode=mode,
        device=device,
    )

    if subset_data:
        warn("Subsetting data to 1000 samples per session.")
        for side in eda_data.keys():
            for user in eda_data[side].keys():
                for session in eda_data[side][user].keys():
                    eda_data[side][user][session] = eda_data[side][user][session][:1000]

    eda_data = {
        side: {
            user: {
                session: make_timestamp_idx(
                    dataframe=session_data,
                    data_name="EDA",
                    individual_name=user,
                    side=side,
                )
                for session, session_data in eda_data[side][user].items()
            }
            for user in eda_data[side].keys()
        }
        for side in eda_data.keys()
    }
    # NOTE: the data here is order this way: {side: {user: session: {Series}}},
    # ir {side: {user: Series}}, depending on the chosen mode.
    # Each pandas Series contains also the `attr` field with the
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

    eda_data_filtered: defaultdict[str, dict[str, ndarray]] = {
        side: {
            user: {
                session_name: Series(
                    butter_lowpass_filter_filtfilt(
                        data=session_data,
                        cutoff=cutoff_frequency,
                        fs=session_data.attrs["sampling frequency"],
                        order=butterworth_order,
                    ),
                    index=session_data.index,
                )
                for session_name, session_data in user_edat_data.items()
            }
            for user, user_edat_data in tqdm(
                eda_data[side].items(), desc=f'Filtering EDA data for side "{side}"'
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
    # eda_data_phasic: defaultdict[str, list[dict[str, ndarray]]] = {
    #     side: {
    #         user: Parallel(n_jobs=n_jobs, backend='threading')(
    #             delayed(decomposition)(
    #                 session_data,
    #                 eda_data[side][user][session].attrs["sampling frequency"],
    #             )
    #             for session, session_data in user_edat_data.items()
    #         )
    #     }
    #     for side in eda_data_filtered.keys()
    #     for user, user_edat_data in eda_data_filtered[side].items()
    # }
    eda_data_phasic: defaultdict[str, list[dict[str, ndarray]]] = {
        side: {
            user: {
                session: Series(
                    decomposition(
                        session_data.values,
                        eda_data[side][user][session].attrs["sampling frequency"],
                        session,
                        user,
                        side,
                    )["phasic component"],
                    index=session_data.index,
                )
                for session, session_data in tqdm(
                    user_edat_data.items(), desc="Session progress"
                )
            }
            for user, user_edat_data in tqdm(
                eda_data_filtered[side].items(),
                desc="EDA decomposition progress (user)",
            )
        }
        for side in eda_data_filtered.keys()
    }

    # eda_data_phasic: defaultdict[str, dict[str, Series]] = {
    #     side: {
    #         user: {
    #             session: Series(
    #                 session_phasic["phasic component"],
    #                 index=eda_data[side][user][session].index,
    #             )
    #             for session, session_phasic in zip(
    #                 eda_data[side][user].keys(), eda_data_phasic[side][user]
    #             )
    #         }
    #         for user in eda_data_phasic[side].keys()
    #     }
    #     for side in eda_data_filtered.keys()
    # }
    print("Total phasic component calculation: %.2f s" % (time() - start))
    if plots:
        make_lineplot(
            data=eda_data_phasic[random_side][random_user][random_session],
            which="EDA",
            savename=f"eda_phasic_{random_side}_{random_user}_{random_session}",
            title="Example EDA phasic component",
        )

    eda_data_standardized: defaultdict[str, dict[str, Series]] = {
        side: {
            user: {
                session: Series(
                    standardize(session_data), index=eda_data[side][user][session].index
                )
                for session, session_data in user_edat_data.items()
            }
            for user, user_edat_data in eda_data_filtered[side].items()
        }
        for side in eda_data_filtered.keys()
    }
    eda_data_standardized_phasic: defaultdict[str, dict[str, dict[str, Series]]] = {
        side: {
            user: {
                session: Series(
                    standardize(session_data), index=eda_data[side][user][session].index
                )
                for session, session_data in user_edat_data.items()
            }
            for user, user_edat_data in eda_data_filtered[side].items()
        }
        for side in eda_data_phasic.keys()
    }

    if plots:
        make_lineplot(
            data=eda_data_standardized[random_side][random_user][random_session],
            which="EDA",
            savename=f"eda_standardized_{random_side}_{random_user}_{random_session}",
            title="Example EDA filtered & standardized",
        )
        make_lineplot(
            data=eda_data_standardized[random_side][random_user][random_session],
            which="EDA",
            savename=f"eda_standardized_{random_side}_{random_user}_{random_session}",
            title="Example EDA phasic standardized",
        )

    if concat_sessions:
        eda_data_standardized_phasic: defaultdict[str, dict[str, Series]] = {
            side: {
                user: concat(
                    [
                        prepare_data_for_concatenation(
                            data=session_data, session_name=session
                        )
                        for session, session_data in user_edat_data.items()
                    ],
                    axis=0,
                    join="outer",
                ).sort_index()
                for user, user_edat_data in eda_data_filtered[side].items()
            }
            for side in eda_data_standardized_phasic.keys()
        }
        eda_data_standardized: defaultdict[str, dict[str, Series]] = {
            side: {
                user: concat(
                    [
                        prepare_data_for_concatenation(
                            data=session_data, session_name=session
                        )
                        for session, session_data in user_edat_data.items()
                    ],
                    axis=0,
                    join="outer",
                ).sort_index()
                for user, user_edat_data in eda_data_filtered[side].items()
            }
            for side in eda_data_standardized.keys()
        }

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
                        user_data_phasic: Series = eda_data_standardized_phasic[side][user]
                        df_to_save: DataFrame = concat(
                            [user_data_standardized, user_data_phasic], axis=1
                        )
                        df_to_save.columns = ["mixed-EDA", "phasic-EDA"]
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
                        filename: str = f"{user}.csv"

                        Path(path_to_save).mkdir(parents=True, exist_ok=True)

                        df_to_save.to_csv(
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
