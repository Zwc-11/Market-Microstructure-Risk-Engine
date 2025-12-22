from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.modeling.train_hazard import train_hazard_model


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_dataframe(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train hazard model.")
    parser.add_argument("--config", required=True, help="Path to config yaml")
    parser.add_argument(
        "--hazard-dataset",
        default=None,
        help="Path to hazard dataset (parquet/csv). Default: artifacts/hazard_dataset.parquet",
    )
    parser.add_argument(
        "--features",
        default=None,
        help="Path to hazard features (parquet/csv). Default: artifacts/hazard_features.parquet",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    artifacts_dir = Path(cfg["paths"]["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    hazard_path = Path(args.hazard_dataset) if args.hazard_dataset else artifacts_dir / "hazard_dataset.parquet"
    features_path = Path(args.features) if args.features else artifacts_dir / "hazard_features.parquet"

    hazard_df = _load_dataframe(hazard_path)
    features_df = _load_dataframe(features_path)

    oof, report = train_hazard_model(hazard_df, features_df, cfg, output_dir=artifacts_dir)

    print("Hazard training report:")
    overall = report.get("overall", {})
    for key in ("auc", "pr_auc", "brier", "lift_top_decile"):
        if key in overall:
            print(f"{key}: {overall[key]}")
    print(f"oof_rows: {len(oof)}")


if __name__ == "__main__":
    main()
