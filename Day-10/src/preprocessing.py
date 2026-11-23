from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "survived"

# Columns that must not be used as model features
DROP_COLUMNS = [
    "alive",
    "class",
    "who",
    "adult_male",
    "deck",
    "embark_town"
]

NUMERIC_FEATURES = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare",
    "alone"
]

CATEGORICAL_FEATURES = [
    "sex",
    "embarked"
]


def prepare_features(df):
    """Separate input features and target."""

    X = df.drop(
        columns=[TARGET_COLUMN] + DROP_COLUMNS,
        errors="ignore"
    )

    y = df[TARGET_COLUMN]

    return X, y


def create_preprocessor():
    """Create numerical and categorical preprocessing pipelines."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES)
        ]
    )

    return preprocessor


if __name__ == "__main__":

    import pandas as pd
    from src.data_loader import load_data

    data = load_data()

    X, y = prepare_features(data)

    preprocessor = create_preprocessor()

    X_transformed = preprocessor.fit_transform(X)

    print("\n========== PREPROCESSING RESULTS ==========")
    print("Original feature shape:", X.shape)
    print("Transformed feature shape:", X_transformed.shape)
    print("Target shape:", y.shape)
    print("Missing values after preprocessing:", 
          pd.DataFrame(X_transformed.toarray() if hasattr(X_transformed, "toarray") else X_transformed).isnull().sum().sum())

    print("\nPreprocessing completed successfully!")
