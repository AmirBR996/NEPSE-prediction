from __future__ import annotations

import csv
import shutil
from pathlib import Path


def count_rows(csv_path: Path) -> int:
    """Return the number of data rows in a CSV file, excluding the header."""
    try:
        with csv_path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.reader(file)
            try:
                next(reader)  # skip header
            except StopIteration:
                return 0
            return sum(1 for _ in reader)
    except FileNotFoundError:
        return 0


def main() -> None:
    root_dir = Path(__file__).resolve().parent
    data_dir = root_dir / "data"
    output_dir = root_dir / "Large_data"

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    output_dir.mkdir(exist_ok=True)

    company_files = sorted(data_dir.glob("*.csv"))
    ranked_companies = []

    for csv_file in company_files:
        row_count = count_rows(csv_file)
        ranked_companies.append((row_count, csv_file.name))

    top_companies = sorted(ranked_companies, key=lambda x: x[0], reverse=True)[:10]

    for row_count, company_name in top_companies:
        source_file = data_dir / company_name
        destination_file = output_dir / company_name
        shutil.copy2(source_file, destination_file)
        print(f"{row_count:>6} rows -> {company_name}")

    summary_path = output_dir / "top_10_companies_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as summary_file:
        writer = csv.writer(summary_file)
        writer.writerow(["company", "row_count"])
        for row_count, company_name in top_companies:
            writer.writerow([company_name, row_count])

    print(f"\nSaved top 10 company files to: {output_dir}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
