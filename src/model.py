
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_SEED


FEATURES = ["age_years", "load_pct", "oil_temp_c"]


def train_and_score():
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "transformers.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} not found. Run `python -m src.data` first."
        )

    df = pd.read_csv(data_path)

    required = FEATURES + ["failed"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df[FEATURES]
    y = df["failed"].astype(int)

    if y.nunique() != 2:
        raise ValueError("The failed column must contain both 0 and 1.")

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=RANDOM_SEED),
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    # Synthetic labels: this AUC only demonstrates that the prediction
    # pipeline works. It is not evidence of real-world predictive accuracy.
    # Every predicted risk comes from a fold whose model did not see that row's label.
    probabilities = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        method="predict_proba",
    )[:, 1]

    df["predicted_risk"] = probabilities

    auc = roc_auc_score(y, probabilities)

    # Refit on all rows for future predictions on new transformers.
    # The existing predicted_risk column remains strictly out-of-fold.
    model.fit(X, y)

    df.to_csv(data_path, index=False)

    print(f"Transformers scored: {len(df)}")
    print(f"Cross-validated ROC-AUC: {auc:.4f}")
    print(f"Mean predicted risk: {df['predicted_risk'].mean():.4f}")
    print(f"Saved scored dataset: {data_path}")
    print("\nTop 10 transformers by predicted risk:")
    print(
        df.nlargest(10, "predicted_risk")[
            ["transformer_id", "age_years", "load_pct",
             "oil_temp_c", "predicted_risk", "nearest_depot"]
        ].to_string(index=False)
    )

    return df, model, auc


if __name__ == "__main__":
    train_and_score()
