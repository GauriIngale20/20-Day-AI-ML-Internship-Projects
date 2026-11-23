
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


NUMERICAL_FEATURES = [
    "pclass", "age", "sibsp", "parch", "fare", "alone"
]

CATEGORICAL_FEATURES = ["sex", "embarked"]


def create_preprocessor():
    """Create a reusable preprocessing pipeline."""

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("numerical", numerical_pipeline, NUMERICAL_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES)
    ])


def create_models():
    """Return the classification models used in the project."""

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=42
        )
    }


def create_model_pipeline(model):
    """Combine preprocessing and classifier into one pipeline."""

    return Pipeline([
        ("preprocessor", create_preprocessor()),
        ("classifier", model)
    ])
