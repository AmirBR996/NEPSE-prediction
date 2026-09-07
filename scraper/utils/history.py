import math
import pandas as pd

_FLOAT_COLS = ["open", "high", "low", "close", "per_change",
               "traded_quantity", "traded_amount"]
_ORDER = ["published_date"] + _FLOAT_COLS + ["status"]


class AuthError(Exception):
    pass


def records_total(resp_json):
    if "recordsTotal" not in resp_json:
        raise AuthError(resp_json.get("message", "missing recordsTotal"))
    return resp_json["recordsTotal"]


def page_starts(total, size):
    pages = math.ceil(total / size)
    return [i * size for i in range(pages)]


def records_to_dataframe(records):
    df = pd.DataFrame.from_dict(records)
    if "DT_Row_Index" in df.columns:
        df = df.drop(columns="DT_Row_Index")
    df = df[::-1].reset_index(drop=True)
    for col in _FLOAT_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["status"] = df["status"].astype(int)
    return df[_ORDER]
