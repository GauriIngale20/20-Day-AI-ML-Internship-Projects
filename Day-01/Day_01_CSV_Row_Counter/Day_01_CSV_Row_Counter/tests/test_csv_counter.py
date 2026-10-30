
from src.csv_counter import count_csv_rows


def test_students_csv():
    assert count_csv_rows("data/students.csv") == 150


def test_employees_csv():
    assert count_csv_rows("data/employees.csv") == 150
