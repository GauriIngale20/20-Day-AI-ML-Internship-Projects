
from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from src.data_loader import load_data, prepare_features_target
from src.preprocessing import create_preprocessor


def main():

    # Load dataset
    df = load_data()
    X, y = prepare_features_target(df)

    # Split before preprocessing to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    numerical_features = ["Age", "SibSp", "Parch", "Fare"]
    categorical_features = ["Pclass", "Sex", "Embarked"]

    # Basic baseline preprocessing: imputation and encoding,
    # without feature scaling
    baseline_numerical = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    baseline_categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    baseline_preprocessor = ColumnTransformer([
        ("num", baseline_numerical, numerical_features),
        ("cat", baseline_categorical, categorical_features)
    ])

    baseline_model = Pipeline([
        ("preprocessor", baseline_preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    # Train baseline
    baseline_model.fit(X_train, y_train)
    baseline_predictions = baseline_model.predict(X_test)

    baseline_accuracy = accuracy_score(
        y_test, baseline_predictions
    )

    # Full preprocessing pipeline with scaling
    full_model = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    full_model.fit(X_train, y_train)
    full_predictions = full_model.predict(X_test)

    full_accuracy = accuracy_score(y_test, full_predictions)

    # Display comparison
    print("\nMODEL ACCURACY COMPARISON")
    print("=" * 40)

    print(f"Baseline Accuracy: {baseline_accuracy:.4f}")
    print(f"Full Pipeline Accuracy: {full_accuracy:.4f}")

    print("\nFULL PIPELINE CLASSIFICATION REPORT")
    print(classification_report(y_test, full_predictions))

    # Save comparison
    results = pd.DataFrame({
        "Model": [
            "Baseline (Imputation + Encoding)",
            "Full Pipeline (Imputation + Encoding + Scaling)"
        ],
        "Accuracy": [
            baseline_accuracy,
            full_accuracy
        ]
    })

    project_dir = Path(__file__).resolve().parent.parent
    reports_dir = project_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    results.to_csv(
        reports_dir / "preprocessing_comparison.csv",
        index=False
    )

    print("\nComparison saved successfully!")
    print("Location: reports/preprocessing_comparison.csv")


if __name__ == "__main__":
    main()

