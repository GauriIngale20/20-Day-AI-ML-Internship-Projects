import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from src.data_loader import load_data
from src.split_data import split_dataset
from src.pipeline import create_models, create_model_pipeline

from config import REPORT_DIR, MODEL_DIR


def train_and_compare_models():

    df = load_data()

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df)

    models = create_models()

    results = []
    trained_models = {}

    print("\n========== MODEL COMPARISON ==========")

    for name, model in models.items():

        pipeline = create_model_pipeline(model)

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_val)

        accuracy = accuracy_score(y_val, predictions)
        precision = precision_score(
            y_val, predictions, zero_division=0
        )
        recall = recall_score(
            y_val, predictions, zero_division=0
        )
        f1 = f1_score(
            y_val, predictions, zero_division=0
        )

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1_Score": f1
        })

        trained_models[name] = pipeline

        print(f"\n{name}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="F1_Score",
        ascending=False
    ).reset_index(drop=True)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(
        REPORT_DIR / "model_comparison.csv",
        index=False
    )

    best_model_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]

    joblib.dump(
        best_model,
        MODEL_DIR / "best_model.joblib"
    )

    print("\n========== FINAL MODEL COMPARISON ==========")
    print(results_df.to_string(index=False))

    print("\nBest model based on validation F1:", best_model_name)
    print("\nModel comparison saved successfully!")
    print("Best model saved successfully!")

    return best_model, results_df


if __name__ == "__main__":
    train_and_compare_models()
