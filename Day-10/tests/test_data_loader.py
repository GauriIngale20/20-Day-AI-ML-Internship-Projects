from src.data_loader import load_data


def test_dataset_loads_successfully():
    df = load_data()

    assert not df.empty
    assert df.shape == (891, 15)


def test_target_column_exists():
    df = load_data()

    assert "survived" in df.columns


def test_target_has_no_missing_values():
    df = load_data()

    assert df["survived"].isnull().sum() == 0
