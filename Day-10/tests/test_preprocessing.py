from src.data_loader import load_data
from src.preprocessing import prepare_features, create_preprocessor
from src.split_data import split_dataset


def test_target_is_separated_from_features():
    df = load_data()

    X, y = prepare_features(df)

    assert "survived" not in X.columns
    assert len(X) == len(y)


def test_leakage_columns_are_removed():
    df = load_data()

    X, y = prepare_features(df)

    leakage_columns = [
        "alive",
        "class",
        "who",
        "adult_male",
        "deck",
        "embark_town"
    ]

    for column in leakage_columns:
        assert column not in X.columns


def test_preprocessing_pipeline_handles_missing_values():
    df = load_data()

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df)

    preprocessor = create_preprocessor()

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    assert X_train_transformed.shape[0] == len(X_train)
    assert X_test_transformed.shape[0] == len(X_test)
    assert X_train_transformed.shape[1] == X_test_transformed.shape[1]
