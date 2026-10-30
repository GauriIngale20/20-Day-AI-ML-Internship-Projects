
import csv
import streamlit as st
from pathlib import Path

from src.csv_counter import count_csv_rows


st.set_page_config(
    page_title="CSV Row Counter",
    page_icon="📊",
    layout="centered"
)

st.title("📊 CSV Row Counter")
st.write("Upload a CSV file or select a sample dataset to count its data rows.")

st.divider()

st.subheader("1. Select a Sample Dataset")

dataset = st.selectbox(
    "Choose a dataset",
    ["Students Dataset", "Employees Dataset"]
)

if dataset == "Students Dataset":
    file_path = Path(__file__).resolve().parent / "data" / "students.csv"
else:
    file_path = Path(__file__).resolve().parent / "data" / "employees.csv"

if st.button("Count Sample Dataset Rows"):
    try:
        total_rows = count_csv_rows(str(file_path))

        st.success("CSV file processed successfully!")

        col1, col2 = st.columns(2)
        col1.metric("Total Data Rows", total_rows)
        col2.metric("Dataset", dataset)

    except (FileNotFoundError, ValueError, OSError) as error:
        st.error(str(error))

st.divider()

st.subheader("2. Upload Your Own CSV File")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        content = uploaded_file.getvalue().decode("utf-8-sig")
        reader = csv.reader(content.splitlines())

        next(reader, None)
        total_rows = sum(1 for row in reader if row)

        st.success("Your CSV file was read successfully!")

        st.metric("Total Data Rows", total_rows)

        st.write("File Name:", uploaded_file.name)

    except UnicodeDecodeError:
        st.error("Unable to read this file. Please upload a UTF-8 CSV file.")
    except csv.Error:
        st.error("Invalid CSV file format.")

st.divider()

st.caption("Developed as part of the 20-Day AI/ML Internship Training.")
