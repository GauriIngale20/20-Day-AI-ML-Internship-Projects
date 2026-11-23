from sklearn.model_selection import train_test_split

from src.data_loader import load_data
from src.preprocessing import prepare_features
from config import RANDOM_STATE


def split_dataset(df):

    X, y = prepare_features(df)

    # First split: 80% development, 20% testing
    X_dev, X_test, y_dev, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Second split: 75% of development for training,
    # 25% of development for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev,
        y_dev,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_dev
    )

    print("\n========== DATA SPLIT RESULTS ==========")

    print("Training:", X_train.shape, y_train.shape)
    print("Validation:", X_val.shape, y_val.shape)
    print("Testing:", X_test.shape, y_test.shape)

    print("\nSurvival Distribution:")

    print("Training:")
    print(y_train.value_counts(normalize=True).round(3))

    print("\nValidation:")
    print(y_val.value_counts(normalize=True).round(3))

    print("\nTesting:")
    print(y_test.value_counts(normalize=True).round(3))

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":

    df = load_data()

    split_dataset(df)

    print("\nData splitting completed successfully!")