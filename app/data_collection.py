import pandas as pd
import os
import fitz  # PyMuPDF for PDFs
from docx import Document
import re

#Collect data from a CSV file
def collect_data(filepath: str):
    if not os.path.exists(filepath):
        raise FileNotFoundError("Data source not found.")
    df = pd.read_csv(filepath)
    return df

#Placeholder function to check permissions (e.g., user authentication)
def verify_permissions():
    return True

# Extract text from a PDF file using PyMuPDF
def extract_from_pdf(filepath: str) -> str:
    if not filepath.endswith('.pdf'):
        raise ValueError("Only PDF files are supported.")
    text = ""
    with fitz.open(filepath) as doc:
        for page in doc:
            text += page.get_text()
    return text

#Extract text from a DOCX file using python-docx
def extract_from_docx(filepath: str) -> str:
    if not filepath.endswith('.docx'):
        raise ValueError("Only DOCX files are supported.")
    doc = Document(filepath)
    return "\n".join([para.text for para in doc.paragraphs])

#Parse text for labeled fields in format "Label: Value"
def parse_labeled_text(text: str) -> dict:
    pattern = re.findall(r'(\w+):\s*(.*)', text)
    return {label: value.strip() for label, value in pattern}

#Wrapper to extract and structure labeled data into a DataFrame
def extract_labeled_data(filepath: str) -> pd.DataFrame:
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.pdf':
        text = extract_from_pdf(filepath)
    elif ext == '.docx':
        text = extract_from_docx(filepath)
    else:
        raise ValueError("Unsupported file format.")

    labeled_dict = parse_labeled_text(text)
    return pd.DataFrame([labeled_dict])  # one-row DataFrame from parsed labels

# Inspect and clean the data
def inspect_data(df: pd.DataFrame):
    # Show top 5 rows of the DataFrame
    print("Data Head:\n", df.head())
    
    # Display schema and types
    print("\nData Info:")
    print(df.info())
    
    # Count missing values in each column
    print("\nMissing Values:\n", df.isnull().sum())
    
    # Show duplicate rows if any
    duplicate_count = df.duplicated().sum()
    print(f"\nDuplicate Rows: {duplicate_count}")

#Clean the data: remove duplicates and rows with missing values
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    df = df.dropna()
    return df
