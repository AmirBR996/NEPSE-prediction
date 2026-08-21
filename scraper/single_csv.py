from __future__ import annotations

import csv
from pathlib import Path


def merge_csv_files(data_dir: Path, output_file: Path) -> None:
	csv_files = sorted(
		path for path in data_dir.glob("*.csv") if path.is_file() and path.name != output_file.name
	)

	if not csv_files:
		raise FileNotFoundError(f"No CSV files found in {data_dir}")

	header_written = False

	with output_file.open("w", newline="", encoding="utf-8") as destination:
		writer = None

		for csv_file in csv_files:
			with csv_file.open("r", newline="", encoding="utf-8-sig") as source:
				reader = csv.reader(source)
				try:
					header = next(reader)
				except StopIteration:
					continue

				if not header_written:
					writer = csv.writer(destination)
					writer.writerow(header)
					header_written = True

				if writer is None:
					writer = csv.writer(destination)

				for row in reader:
					if row:
						writer.writerow(row)

	if not header_written:
		raise ValueError(f"CSV files in {data_dir} did not contain any rows")


def main() -> None:
	root_dir = Path(__file__).resolve().parent.parent
	data_dir = root_dir / "data"
	output_file = root_dir / "data.csv"

	merge_csv_files(data_dir, output_file)
	print(f"Created {output_file}")


if __name__ == "__main__":
	main()
