from pathlib import Path

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


# Project paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "raw" / "titanic.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_and_prepare_data():

    print("\nLoading dataset...")

    df = pd.read_csv(DATA_PATH)

    # Remove unnecessary columns
    X = df.drop(
        columns=[
            "Survived",
            "PassengerId",
            "Name",
            "Ticket",
            "Cabin"
        ]
    )

    y = df["Survived"]

    # Numerical and categorical columns
    numerical_features = [
        "Pclass",
        "Age",
        "SibSp",
        "Parch",
        "Fare"
    ]

    categorical_features = [
        "Sex",
        "Embarked"
    ]

    # Numerical preprocessing
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical preprocessing
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine preprocessing
    preprocessor = ColumnTransformer([
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    return X, y, preprocessor


def train_models():

    X, y, preprocessor = load_and_prepare_data()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\n========== DATA SPLIT ==========")

    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))

    print("\nTraining class distribution:")
    print(y_train.value_counts())

    print("\nTesting class distribution:")
    print(y_test.value_counts())

    # Define models
    models = {

        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

        "balanced_logistic_regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ),

        "decision_tree": DecisionTreeClassifier(
            max_depth=5,
            random_state=42
        )
    }

    print("\n========== MODEL TRAINING ==========")

    for name, classifier in models.items():

        print(f"\nTraining {name}...")

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ])

        pipeline.fit(X_train, y_train)

        # Save trained model
        model_path = MODEL_DIR / f"{name}.pkl"

        joblib.dump(pipeline, model_path)

        print(f"{name} trained successfully!")
        print(f"Model saved at: {model_path}")

    # Save the same test data for all models
    test_data = {
        "X_test": X_test,
        "y_test": y_test
    }

    joblib.dump(
        test_data,
        MODEL_DIR / "test_data.pkl"
    )

    print("\nTest data saved successfully!")

    print("\n========== TRAINING COMPLETED ==========")


if __name__ == "__main__":
    train_models()
