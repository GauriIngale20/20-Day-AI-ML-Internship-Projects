from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from src.data_loader import load_data
from src.preprocessing import create_preprocessor
from src.split_data import split_dataset

from config import RANDOM_STATE


def test_model_pipeline_trains_and_predicts():

    df = load_data()

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df)

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            ("model", LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE
            ))
        ]
    )

    model_pipeline.fit(X_train, y_train)

    predictions = model_pipeline.predict(X_val)

    assert len(predictions) == len(y_val)
    assert set(predictions).issubset({0, 1})
