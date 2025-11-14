"""
Small script to train a minimal LogisticRegression model on the example CSV
and save a recommended dict: {'model', 'scaler', 'feature_columns'} to
`example_model.pkl` using joblib.

Run in the project folder: python generate_example_model.py
"""
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

# Load example dataset
df = pd.read_csv('example_dataset.csv')
# Auto-detect target column
target_candidates = [c for c in df.columns if c.lower() in ('label','attack','target','class')]
if not target_candidates:
    raise SystemExit('No target column found in example_dataset.csv')

target = target_candidates[0]
X = df.drop(columns=[target]).select_dtypes(include=['number'])
y = df[target]

# Simple preprocessing and train
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X.fillna(0))
clf = LogisticRegression(max_iter=1000)
clf.fit(X_scaled, y)

# Save recommended structure
artifact = {'model': clf, 'scaler': scaler, 'feature_columns': list(X.columns)}
joblib.dump(artifact, 'example_model.pkl')
print('Saved example_model.pkl')
