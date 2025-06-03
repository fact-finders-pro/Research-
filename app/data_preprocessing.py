# ========================================
# IMPORTS
# ========================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import faiss

# ========================================
# GENERATE EMBEDDINGS FROM TEXT DATA
# ========================================
def generate_embeddings(df: pd.DataFrame, text_column='text'):
    """
    Generate vector embeddings from a DataFrame column using a pre-trained SentenceTransformer.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Pretrained model for sentence embeddings
    embeddings = model.encode(df[text_column].tolist(), show_progress_bar=True)
    return embeddings

# ========================================
# BUILD FAISS VECTOR INDEX FOR SIMILARITY SEARCH
# ========================================
def build_faiss_index(embeddings: np.ndarray):
    """
    Build a FAISS index using L2 (Euclidean) distance from a NumPy array of embeddings.
    """
    dim = embeddings.shape[1]  # e.g., 384 for MiniLM
    index = faiss.IndexFlatL2(dim)  # Flat (exact) index using L2 distance
    index.add(embeddings)  # Add all embedding vectors to the index
    return index

# ========================================
# SEARCH FAISS INDEX
# ========================================
def search_index(index, query_text: str, model, df: pd.DataFrame, k=5):
    """
    Search the FAISS index using a query string and return top-k most similar entries.
    """
    query_embedding = model.encode([query_text])
    distances, indices = index.search(np.array(query_embedding), k)

    results = []
    for i, idx in enumerate(indices[0]):
        result = {
            "rank": i + 1,
            "matched_text": df.iloc[idx]['text'],
            "distance": distances[0][i]
        }
        results.append(result)
    
    return results

# ========================================
# SPLIT DATA INTO TRAIN, VALIDATION, TEST
# ========================================
def split_data(X, y):
    """
    Split data into training, validation, and test sets (70/15/15).
    """
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

# ========================================
#EXAMPLE USAGE
# ========================================
if __name__ == "__main__":
    # Sample text data
    data = {
        "text": [
            "Machine learning is fascinating.",
            "Natural language processing with transformers.",
            "Deep learning advances AI.",
            "The cat sat on the mat.",
            "Neural networks are powerful models."
        ]
    }

    df = pd.DataFrame(data)

    # Generate embeddings
    embeddings = generate_embeddings(df)

    # Build FAISS index
    index = build_faiss_index(embeddings)

    # Load model again for querying (or reuse)
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Search example
    query = "What is deep learning?"
    results = search_index(index, query, model, df, k=3)

    print("\n🔍 Top 3 Matches for Query:")
    for res in results:
        print(f"{res['rank']}. {res['matched_text']} (distance: {res['distance']:.4f})")


