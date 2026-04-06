from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASELINE_NUMERIC = ["temp", "hour_of_day", "atemp"]
BASELINE_CATEGORICAL = ["season", "workingday"]

IMPROVED_NUMERIC = ["temp", "atemp", "rhum", "wspd", "hour_of_day", "day_of_week", "is_weekend", "holiday", "workingday"]
IMPROVED_CATEGORICAL = ["season", "time_of_day"]

TARGET = "total_rentals"


def validate_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def build_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
    model,
) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def time_ordered_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    split_idx = int(len(X) * (1 - test_size))
    if split_idx <= 0 or split_idx >= len(X):
        raise ValueError("Invalid split index. Adjust --test-size so both train and test sets are non-empty.")

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    return X_train, X_test, y_train, y_test


def add_cv_metrics(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    folds: int,
) -> Dict[str, float]:
    tscv = TimeSeriesSplit(n_splits=folds)
    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=tscv,
        scoring={
            "neg_rmse": "neg_root_mean_squared_error",
            "neg_mae": "neg_mean_absolute_error",
            "r2": "r2",
        },
        n_jobs=-1,
    )

    return {
        "cv_rmse_mean": float(-scores["test_neg_rmse"].mean()),
        "cv_rmse_std": float(scores["test_neg_rmse"].std()),
        "cv_mae_mean": float(-scores["test_neg_mae"].mean()),
        "cv_mae_std": float(scores["test_neg_mae"].std()),
        "cv_r2_mean": float(scores["test_r2"].mean()),
        "cv_r2_std": float(scores["test_r2"].std()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train bike demand baseline and improved models.")
    parser.add_argument(
        "--input-csv",
        type=str,
        default="data/hourly_bike_weather_2017.csv",
        help="Joined hourly dataset produced by pipeline.py.",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="outputs/models",
        help="Directory to store trained model artifacts.",
    )
    parser.add_argument(
        "--metrics-json",
        type=str,
        default="outputs/tables/modeling/model_metrics.json",
        help="Path for model metrics JSON.",
    )
    parser.add_argument(
        "--metrics-csv",
        type=str,
        default="outputs/tables/modeling/model_metrics.csv",
        help="Path for model metrics table CSV.",
    )
    parser.add_argument(
        "--predictions-csv",
        type=str,
        default="outputs/tables/modeling/test_predictions.csv",
        help="Path for test-set predictions CSV.",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default="outputs/figures",
        help="Directory for model diagnostics figures.",
    )
    parser.add_argument(
        "--residuals-csv",
        type=str,
        default="outputs/tables/modeling/model_residuals.csv",
        help="Path for residual diagnostics CSV.",
    )
    parser.add_argument(
        "--feature-importance-csv",
        type=str,
        default="outputs/tables/modeling/random_forest_feature_importance.csv",
        help="Path for random forest feature importance CSV.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split ratio (default 0.2 means 80/20 split).",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of time-series CV folds for training diagnostics.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_csv = Path(args.input_csv)
    models_dir = Path(args.models_dir)
    metrics_json = Path(args.metrics_json)
    metrics_csv = Path(args.metrics_csv)
    predictions_csv = Path(args.predictions_csv)
    residuals_csv = Path(args.residuals_csv)
    importance_csv = Path(args.feature_importance_csv)
    figures_dir = Path(args.figures_dir)

    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_json.parent.mkdir(parents=True, exist_ok=True)
    predictions_csv.parent.mkdir(parents=True, exist_ok=True)
    residuals_csv.parent.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    df = pd.read_csv(input_csv, parse_dates=["hour"])
    df = df.sort_values("hour").reset_index(drop=True)

    required = list(set(BASELINE_NUMERIC + BASELINE_CATEGORICAL + IMPROVED_NUMERIC + IMPROVED_CATEGORICAL + [TARGET]))
    validate_columns(df, required)

    y = df[TARGET]
    X = df.drop(columns=[TARGET])

    X_train, X_test, y_train, y_test = time_ordered_split(
        X=X,
        y=y,
        test_size=args.test_size,
    )

    baseline_model = build_pipeline(
        numeric_features=BASELINE_NUMERIC,
        categorical_features=BASELINE_CATEGORICAL,
        model=LinearRegression(),
    )
    baseline_model.fit(X_train, y_train)
    baseline_pred = baseline_model.predict(X_test)
    baseline_metrics = evaluate_predictions(y_test, baseline_pred)
    baseline_metrics.update(add_cv_metrics(baseline_model, X_train, y_train, folds=args.cv_folds))

    improved_model = build_pipeline(
        numeric_features=IMPROVED_NUMERIC,
        categorical_features=IMPROVED_CATEGORICAL,
        model=RandomForestRegressor(
            n_estimators=300,
            random_state=args.random_state,
            n_jobs=-1,
        ),
    )
    improved_model.fit(X_train, y_train)
    improved_pred = improved_model.predict(X_test)
    improved_metrics = evaluate_predictions(y_test, improved_pred)
    improved_metrics.update(add_cv_metrics(improved_model, X_train, y_train, folds=args.cv_folds))

    metrics = {
        "baseline_linear_regression": baseline_metrics,
        "improved_random_forest": improved_metrics,
    }

    with metrics_json.open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    metrics_table = pd.DataFrame(metrics).T.reset_index().rename(columns={"index": "model"})
    metrics_table.to_csv(metrics_csv, index=False)

    prediction_table = pd.DataFrame(
        {
            "hour": X_test["hour"],
            "actual": y_test,
            "baseline_prediction": baseline_pred,
            "improved_prediction": improved_pred,
        }
    ).sort_values("hour")
    prediction_table.to_csv(predictions_csv, index=False)

    residual_table = prediction_table.copy()
    residual_table["baseline_residual"] = residual_table["actual"] - residual_table["baseline_prediction"]
    residual_table["improved_residual"] = residual_table["actual"] - residual_table["improved_prediction"]
    residual_table.to_csv(residuals_csv, index=False)

    dump(baseline_model, models_dir / "baseline_linear_regression.joblib")
    dump(improved_model, models_dir / "improved_random_forest.joblib")

    preprocessor = improved_model.named_steps["preprocessor"]
    rf = improved_model.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    importance_table = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": rf.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance_table.to_csv(importance_csv, index=False)

    window = prediction_table.head(24 * 14)  # first 2 weeks of test set

    plt.figure(figsize=(11, 5))
    plt.plot(window["hour"], window["actual"], label="Actual", linewidth=1.5)
    plt.plot(window["hour"], window["baseline_prediction"], label="Baseline", linewidth=1)
    plt.plot(window["hour"], window["improved_prediction"], label="Improved", linewidth=1)
    plt.title("Actual vs Predicted Rentals — First 2 Weeks of Test Window (Hourly)")
    plt.xlabel("Hour")
    plt.ylabel("Total Rentals")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "model_actual_vs_predicted.png", dpi=220, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=residual_table,
        x="improved_prediction",
        y="improved_residual",
        s=20,
        alpha=0.45,
    )
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
    plt.title("Improved Model Residuals vs Predictions")
    plt.xlabel("Predicted Rentals")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.tight_layout()
    plt.savefig(figures_dir / "model_residuals_vs_predicted.png", dpi=220, bbox_inches="tight")
    plt.close()

    print("Training complete.")
    print("Baseline metrics:", baseline_metrics)
    print("Improved metrics:", improved_metrics)
    print(f"Saved metrics to: {metrics_json} and {metrics_csv}")
    print(f"Saved predictions to: {predictions_csv}")
    print(f"Saved residual diagnostics to: {residuals_csv}")
    print(f"Saved model artifacts to: {models_dir}")
    print(f"Saved random forest feature importances to: {importance_csv}")
    print(f"Saved model figures to: {figures_dir}")


if __name__ == "__main__":
    main()