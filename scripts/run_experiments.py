import argparse
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def build_preprocessor(numeric_cols, categorical_cols, scaler):
    num_pipe = Pipeline([("scaler", scaler)])
    cat_pipe = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
    )


def evaluate_models(df: pd.DataFrame, quick: bool = False) -> pd.DataFrame:
    target_col = "Drug"
    if target_col not in df.columns:
        raise ValueError(f"Expected target column '{target_col}' in dataset.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Detect columns by dtype (works even if you changed encoding in notebook)
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scalers = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
        "maxabs": MaxAbsScaler(),
    }

    models = {
        "logreg": LogisticRegression(max_iter=2000),
        "knn": KNeighborsClassifier(n_neighbors=5),
        "svm": SVC(),
        "decision_tree": DecisionTreeClassifier(random_state=42),
    }

    if quick:
        # faster CI/local sanity check
        scalers = {"standard": StandardScaler()}
        models = {"logreg": LogisticRegression(max_iter=2000)}

    rows = []
    for scaler_name, scaler in scalers.items():
        pre = build_preprocessor(numeric_cols, categorical_cols, scaler)

        for model_name, model in models.items():
            pipe = Pipeline([("preprocess", pre), ("model", model)])
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)

            rows.append(
                {
                    "model": model_name,
                    "scaler": scaler_name,
                    "accuracy": accuracy_score(y_test, preds),
                    "precision_macro": precision_score(y_test, preds, average="macro", zero_division=0),
                    "recall_macro": recall_score(y_test, preds, average="macro", zero_division=0),
                    "f1_macro": f1_score(y_test, preds, average="macro", zero_division=0),
                }
            )

    return pd.DataFrame(rows).sort_values("f1_macro", ascending=False).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/drug200.csv")
    parser.add_argument("--out", default="reports/results.csv")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    results = evaluate_models(df, quick=args.quick)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)

    print("Top results (by F1-macro):")
    print(results.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
