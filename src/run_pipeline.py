"""
Run the whole offline pipeline end to end:

    raw CSVs -> clean tables -> features -> model -> database

Usage:
    python -m src.run_pipeline
"""

import time

from src import build_database, data_loader, feature_engineering, train_model


def main() -> None:
    t0 = time.perf_counter()
    for stage in (data_loader, feature_engineering, train_model, build_database):
        stage.run()
        print()
    print(f"Pipeline finished in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
