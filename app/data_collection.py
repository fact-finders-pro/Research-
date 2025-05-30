import pandas as pd
import os

def collect_data(filepath: str):
if not os.path.exists(filepath):
raise FileNotFoundError("Data source not found.")
df = pd.read_csv(filepath)
return df

def verify_permissions():
# Placeholder for permission check logic
return True

def inspect_data(df: pd.DataFrame):
print("Data Head:", df.head())
print("Data Info:")
print(df.info())
print("Missing Values:", df.isnull().sum()) 