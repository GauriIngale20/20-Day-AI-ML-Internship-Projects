
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


def create_preprocessor():
    """Create preprocessing pipeline for Titanic dataset."""

    numerical_features = [
        "Pclass", "Age", "SibSp", "Parch", "Fare"
    ]

    categorical_features = [
        "Sex", "Embarked"
    ]

    # Numerical data preprocessing
    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical data preprocessing
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine both pipelines
    preprocessor = ColumnTransformer(transformers=[
        ("numerical", numerical_pipeline, numerical_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    return preprocessor
