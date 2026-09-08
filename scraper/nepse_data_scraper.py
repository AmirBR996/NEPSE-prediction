import time
import requests
import sys
from pathlib import Path

import pandas as pd

SCRAPER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRAPER_DIR.parent

if str(SCRAPER_DIR) not in sys.path:
    sys.path.insert(0, str(SCRAPER_DIR))

from constants.companyIdMap import companyIdMap
from constants.url import historyUrl
from utils.session import make_session, prime_session, TIMEOUT
from utils.params import build_history_payload
from utils.history import page_starts, records_to_dataframe, records_total, AuthError

OUT_DIR = PROJECT_ROOT / "data"
PAGE_SIZE = 50
DELAY = 0.3
MAX_ATTEMPTS = 4


def _post(session, token, company_id, start, length):
    time.sleep(DELAY)
    payload = build_history_payload(start, length, company_id, token)
    resp = session.post(historyUrl, data=payload, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def _load_existing_rows(out_file):
    if not out_file.exists():
        return None
    return pd.read_csv(out_file)


def get_last_date(existing_df):
    if existing_df is None or existing_df.empty:
        return None

    dates = pd.to_datetime(
        existing_df["published_date"],
        errors="coerce"
    ).dropna()

    if dates.empty:
        return None

    return dates.max()


def fetch_new_history(session, token, company_id, last_date, size=PAGE_SIZE):
    rows = []
    start = 0

    if last_date is None:
        first = _post(session, token, company_id, 0, 1)
        total = records_total(first)

        for start in page_starts(total, size):
            page = _post(session, token, company_id, start, size)
            rows.extend(page.get("data", []))

        return rows

    while True:
        page = _post(session, token, company_id, start, size)
        page_rows = page.get("data", [])

        if not page_rows:
            break

        rows.extend(page_rows)

        page_df = records_to_dataframe(page_rows)

        if "published_date" not in page_df.columns:
            break

        page_dates = pd.to_datetime(
            page_df["published_date"],
            errors="coerce"
        ).dropna()

        if not page_dates.empty:
            oldest_date = page_dates.min()

            if oldest_date <= last_date:
                break

        start += size

    if not rows:
        return []

    new_df = records_to_dataframe(rows)

    if new_df.empty:
        return []

    new_df["published_date"] = pd.to_datetime(
        new_df["published_date"],
        errors="coerce"
    )

    new_df = new_df[
        new_df["published_date"] > last_date
    ]

    return new_df.to_dict("records")


def collect_company(session, token, symbol, company_id, last_date):
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            print(
                f"Collecting {symbol} "
                f"(id={company_id}) "
                f"[attempt {attempt}]..."
            )

            return (
                fetch_new_history(
                    session,
                    token,
                    company_id,
                    last_date
                ),
                token
            )

        except (AuthError, requests.RequestException) as exc:
            print(
                f"  issue ({type(exc).__name__}): {exc}; "
                f"refreshing token"
            )

            time.sleep(2)

            try:
                token = prime_session(session)
            except requests.RequestException:
                pass

        except Exception as exc:
            print(f"  FAILED {symbol}: {exc}")
            return None, token

    return None, token


def _merge_and_write(out_file, existing_df, fresh_df):
    if existing_df is None or existing_df.empty:
        combined = fresh_df.copy()
    else:
        combined = pd.concat(
            [existing_df, fresh_df],
            ignore_index=True
        )

    combined["published_date"] = pd.to_datetime(
        combined["published_date"],
        errors="coerce"
    )

    combined = combined.dropna(
        subset=["published_date"]
    )

    combined = combined.sort_values(
        "published_date"
    )

    combined = combined.drop_duplicates(
        subset=["published_date"],
        keep="last"
    )

    combined["published_date"] = combined[
        "published_date"
    ].dt.strftime("%Y-%m-%d")

    combined = combined[
        [
            "published_date",
            "open",
            "high",
            "low",
            "close",
            "per_change",
            "traded_quantity",
            "traded_amount",
            "status",
        ]
    ]

    combined.to_csv(
        out_file,
        index=False
    )

    return combined


def main():
    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    session = make_session()
    token = prime_session(session)

    seeded = 0
    updated = 0
    empty = 0
    failed = 0

    for symbol, company_id in companyIdMap.items():

        out_file = OUT_DIR / f"{symbol}.csv"

        existing_df = _load_existing_rows(
            out_file
        )

        last_date = get_last_date(
            existing_df
        )

        if last_date is None:
            print(
                f"\n{symbol}: no existing data, "
                f"fetching full history"
            )
        else:
            print(
                f"\n{symbol}: latest stored date = "
                f"{last_date.strftime('%Y-%m-%d')}"
            )

        rows, token = collect_company(
            session,
            token,
            symbol,
            company_id,
            last_date
        )

        if rows is None:
            failed += 1
            continue

        if not rows:
            print(
                f"  {symbol}: no new records"
            )
            empty += 1
            continue

        fresh_df = records_to_dataframe(
            rows
        )

        merged_df = _merge_and_write(
            out_file,
            existing_df,
            fresh_df
        )

        if existing_df is None:
            seeded += 1

            print(
                f"  wrote {len(merged_df)} rows "
                f"-> {out_file.name}"
            )

        else:
            updated += 1

            added = (
                len(merged_df)
                - len(existing_df)
            )

            if added > 0:
                print(
                    f"  updated {out_file.name}: "
                    f"+{added} rows"
                )
            else:
                print(
                    f"  {out_file.name} "
                    f"already up to date"
                )

    print(
        f"\nDone. "
        f"seeded={seeded} "
        f"updated={updated} "
        f"empty={empty} "
        f"failed={failed}"
    )


if __name__ == "__main__":
    main()