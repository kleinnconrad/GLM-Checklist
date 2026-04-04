import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

# Generate dummy time-series data
dates = pd.date_range(start='2023-01-01', periods=1000, freq='D')
data = pd.DataFrame({
    'date': dates,
    'feature': np.random.normal(0, 1, 1000),
    'target': np.random.randint(0, 2, 1000)
})

print("\n--- 4. Training & Validation ---")

# 1. Out-of-Time Validation (Strict temporal shift)
train_data = data[data['date'] < '2025-01-01']
val_data = data[(data['date'] >= '2025-01-01') & (data['date'] < '2025-08-01')]
# Prediction/Test set would be > 2025-08-01

X_train, y_train = train_data[['feature']], train_data['target']
X_val, y_val = val_data[['feature']], val_data['target']

# Train Model
model = LogisticRegression().fit(X_train, y_train)

# 2. Overfitting / Underfitting Check
train_preds = model.predict_proba(X_train)[:, 1]
val_preds = model.predict_proba(X_val)[:, 1]

train_loss = log_loss(y_train, train_preds)
val_loss = log_loss(y_val, val_preds)

print(f"Training Log-Loss: {train_loss:.4f}")
print(f"Validation Log-Loss: {val_loss:.4f}")
if val_loss > train_loss * 1.2:
    print("Warning: Potential Overfitting detected.")
else:
    print("Model generalizes well to the validation set.")
  
