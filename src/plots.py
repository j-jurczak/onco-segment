"""
Code to create visualizations
"""

from pathlib import Path

from src.config import FIGURES_DIR, PROCESSED_DATA_DIR


def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = FIGURES_DIR / "plot.png",
    # -----------------------------------------
):

    pass


if __name__ == "__main__":
    main()
