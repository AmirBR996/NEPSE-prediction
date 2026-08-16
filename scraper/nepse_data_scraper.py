import time
import requests
from pathlib import Path

import pandas as pd

from constants.companyIdMap import companyIdMap
from constants.url import historyUrl
from utils.session import make_session, prime_session, TIMEOUT
from utils.params import build_history_payload
from utils.history import page_starts, records_to_dataframe, records_total, AuthError

OUT_DIR = Path("../data/")
PAGE_SIZE = 50  
DELAY = 0.3  
MAX_ATTEMPTS = 4  


def _post(session, token, company_id, start, length):
    time.sleep(DELAY)
    payload = build_history_payload(start, length, company_id, token)
    resp = session.post(historyUrl, data=payload, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def fetch_history(session, token, company_id, size=PAGE_SIZE):
    """Fetch all history rows for a company (newest-first). Raises AuthError if
    the API returns an error payload (caller refreshes the token and retries)."""
    first = _post(session, token, company_id, 0, 1)
    total = records_total(first)
    rows = []
    for start in page_starts(total, size):
        page = _post(session, token, company_id, start, size)
        records_total(page)  
        rows.extend(page.get("data", []))
    return rows


def collect_company(session, token, symbol, company_id):
    """Return (rows, token). rows is None on failure, [] if genuinely untraded.
    Refreshes the CSRF token and retries on auth/HTTP/timeout errors."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            print(f"Collecting {symbol} (id={company_id}) [attempt {attempt}]...")
            return fetch_history(session, token, company_id), token
        except (AuthError, requests.RequestException) as exc:
            print(f"  issue ({type(exc).__name__}): {exc}; refreshing token")
            time.sleep(2)
            try:
                token = prime_session(session)
            except requests.RequestException:
                pass
        except Exception as exc:  
            print(f"  FAILED {symbol}: {exc}")
            return None, token
    return None, token


def _load_existing_rows(out_file):
    if not out_file.exists():
        return None
    return pd.read_csv(out_file)


def _merge_and_write(out_file, existing_df, fresh_df):
    if existing_df is None or existing_df.empty:
        combined = fresh_df.copy()
    else:
        combined = pd.concat([existing_df, fresh_df], ignore_index=True)

    combined["published_date"] = pd.to_datetime(combined["published_date"], errors="coerce")
    combined = combined.dropna(subset=["published_date"])
    combined = combined.sort_values("published_date")
    combined = combined.drop_duplicates(subset=["published_date"], keep="last")
    combined["published_date"] = combined["published_date"].dt.strftime("%Y-%m-%d")
    combined = combined[[
        "published_date",
        "open",
        "high",
        "low",
        "close",
        "per_change",
        "traded_quantity",
        "traded_amount",
        "status",
    ]]
    combined.to_csv(out_file, index=False)
    return combined


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    session = make_session()
    token = prime_session(session)

    seeded = updated = empty = failed = 0
    for symbol, company_id in companyIdMap.items():
        out_file = OUT_DIR / f"{symbol}.csv"
        existing_df = _load_existing_rows(out_file)

        rows, token = collect_company(session, token, symbol, company_id)
        if rows is None:
            failed += 1
            continue
        if not rows:
            print(f"  no records for {symbol} (untraded)")
            empty += 1
            continue

        fresh_df = records_to_dataframe(rows)
        merged_df = _merge_and_write(out_file, existing_df, fresh_df)

        if existing_df is None:
            seeded += 1
            print(f"  wrote {len(merged_df)} rows -> {out_file.name}")
        else:
            updated += 1
            added = len(merged_df) - len(existing_df)
            if added > 0:
                print(f"  updated {out_file.name}: +{added} rows")
            else:
                print(f"  {out_file.name} already up to date")

    print(f"\nDone. seeded={seeded} updated={updated} empty={empty} failed={failed}")


if __name__ == "__main__":
    main()
