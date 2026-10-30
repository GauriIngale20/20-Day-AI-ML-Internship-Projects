
# CSV Row Counter

A Python-based CSV Row Counter application that reads CSV files and counts the total number of data rows.

## Features

- Count rows in CSV files.
- Supports student and employee datasets.
- Upload and count custom CSV files.
- Command Line Interface (CLI).
- Streamlit web interface.
- Error handling for missing or invalid files.
- Unit testing using Pytest.

## Technologies Used

- Python
- Streamlit
- CSV Module
- Pytest
- Git and GitHub

## Project Structure

```text
Day_01_CSV_Row_Counter/
├── data/
├── src/
├── tests/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## How to Run

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run CLI application

```bash
python -m src.main data/students.csv
```

### 4. Run Streamlit application

```bash
streamlit run app.py
```

### 5. Run tests

```bash
python -m pytest -v
```

## Datasets

- Student Dataset: 150 records
- Employee Dataset: 150 records

## Developed By

Gauri Ingale
