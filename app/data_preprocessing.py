import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def clean_data(df: pd.DataFrame):
df = df.drop_duplicates()
df = df.dropna()
return df

def preprocess_features(df: pd.DataFrame):
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df)
return scaled_features

def split_data(X, y):
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
return X_train, X_val, X_test, y_train, y_val, y_test
