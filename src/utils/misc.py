from pandas import Series, DataFrame
from numpy import ndarray


def get_all_users(data: dict[str, dict[str, Series | DataFrame]]):
    users_left = data["left"].keys()
    users_right = data["right"].keys()
    return list(set(users_left) & set(users_right))


def get_all_users_sessions(
    data: dict[str, dict[str, Series | DataFrame]]
) -> list[tuple[str, str]]:
    users = get_all_users(data)
    results = []
    for user in sorted(users):
        sessions_all_left = sorted(
            data["left"][user].index.get_level_values(0).unique()
        )
        sessions_all_right = sorted(
            data["right"][user].index.get_level_values(0).unique()
        )
        sessions_all = sorted(set(sessions_all_left) & set(sessions_all_right))
        for session in sessions_all:
            results.append((user, session))
    return results


def get_all_sessions(
    user_data_left: DataFrame | dict[str, DataFrame],
    user_data_right: DataFrame | dict[str, DataFrame],
) -> list[str]:
    
    if isinstance(user_data_left, DataFrame) and isinstance(user_data_right, DataFrame):
        sessions_all_left: list[str] = sorted(user_data_left.index.get_level_values(0).unique())
        sessions_all_right: list[str] = sorted(user_data_right.index.get_level_values(0).unique())
    elif isinstance(user_data_left, dict) and isinstance(user_data_right, dict):
        sessions_all_left: list[str] = sorted(user_data_left.keys())
        sessions_all_right: list[str] = sorted(user_data_right.keys())
    else:
        raise RuntimeError(f"Invalid data type. Data types are {type(user_data_left)=} and {type(user_data_right)=}.")
    sessions_all = sorted(set(sessions_all_left) & set(sessions_all_right))
    return sessions_all


def get_labels_counts(
    labels_left: ndarray,
    labels_right: ndarray,
    events: list[str] = ["relaxation", "cognitive load"],
) -> DataFrame:
    counts = {
        "left": Series(labels_left).value_counts(),
        "right": Series(labels_right).value_counts(),
    }
    counts = DataFrame(counts).unstack().reset_index()
    counts.columns = ["side", "label", "count"]
    counts["label"] = counts["label"].map({0: events[0], 1: events[1]})
    return counts
