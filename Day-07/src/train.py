
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from src.data_loader import load_data
from src.preprocessing import create_preprocessor


def train_models():

    df = load_data()

    features = [
        "Pclass", "Sex", "Age", "SibSp",
        "Parch", "Fare", "Embarked"
    ]

    X = df[features]
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    logistic_model = Pipeline(steps=[
        ("preprocessor", create_preprocessor()),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    tree_model = Pipeline(steps=[
        ("preprocessor", create_preprocessor()),
        ("classifier", DecisionTreeClassifier(
            max_depth=5,
            random_state=42
        ))
    ])

    logistic_model.fit(X_train, y_train)
    tree_model.fit(X_train, y_train)

    # Save trained models
    project_root = Path(__file__).resolve().parent.parent
    models_folder = project_root / "models"
    models_folder.mkdir(exist_ok=True)

    joblib.dump(
        logistic_model,
        models_folder / "logistic_regression.pkl"
    )

    joblib.dump(
        tree_model,
        models_folder / "decision_tree.pkl"
    )

    print("\nBoth models trained and saved successfully!")

    print("Training data shape:", X_train.shape)
    print("Testing data shape:", X_test.shape)

    return logistic_model, tree_model, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    train_models()

