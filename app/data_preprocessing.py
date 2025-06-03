import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def clean_data(df: pd.DataFrame):
    df = df.drop_duplicates()
    df = df.dropna()
    return df

def generate_embeddings(df: pd.DataFrame, text_column='text'):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(df[text_column].tolist(), show_progress_bar=True)

def build_faiss_index(embeddings: np.ndarray):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def split_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

