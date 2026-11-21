
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from src.train import train_models


def evaluate_models():

    # Train models and get test data
    logistic_model, tree_model, X_train, X_test, y_train, y_test = train_models()

    models = {
        "Logistic Regression": logistic_model,
        "Decision Tree": tree_model
    }

    # 5-Fold Cross-Validation
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    results = []

    for name, model in models.items():

        print("\n" + "=" * 50)
        print(name)
        print("=" * 50)

        # Predictions
        y_pred = model.predict(X_test)

        # Evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # Training accuracy to understand overfitting
        train_accuracy = model.score(X_train, y_train)

        # 5-fold cross-validation on training data
        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="accuracy"
        )

        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Testing Accuracy:  {accuracy:.4f}")
        print(f"Precision:         {precision:.4f}")
        print(f"Recall:            {recall:.4f}")
        print(f"F1 Score:          {f1:.4f}")
        print(f"5-Fold CV Scores:  {cv_scores}")
        print(f"Mean CV Accuracy:  {cv_scores.mean():.4f}")

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        results.append({
            "Model": name,
            "Training Accuracy": train_accuracy,
            "Testing Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Mean CV Accuracy": cv_scores.mean()
        })

    # Final comparison
    results_df = pd.DataFrame(results)

    print("\n\nFINAL MODEL COMPARISON")
    print(results_df.to_string(index=False))

    return results_df


if __name__ == "__main__":
    evaluate_models()
