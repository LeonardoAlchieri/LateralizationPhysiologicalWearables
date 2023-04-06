from pandas import Series, DataFrame

def get_all_users(data: dict[str, dict[str, Series | DataFrame]]):
    users_left = data["left"].keys()
    users_right = data["right"].keys()
    return list(set(users_left) & set(users_right))