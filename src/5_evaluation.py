import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve

# Generate dummy predictions and ground truth
np.random.seed(42)
y_true = np.random.randint(0, 2, 1000)
y_prob = np.random.beta(2, 5, 1000) # Simulated probabilities

print("\n--- 5. Model Evaluation & Business Metrics ---")

# 1. Optimal Cutoff Point (via KS Statistic / ROC Curve)
fpr, tpr, thresholds = roc_curve(y_true, y_prob)
ks_stat = tpr - fpr
optimal_idx = np.argmax(ks_stat)
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal Cutoff Threshold (Max KS): {optimal_threshold:.4f}")

# 2. Alpha & Beta Error Analysis (Confusion Matrix)
y_pred_optimal = (y_prob >= optimal_threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred_optimal).ravel()
print(f"Confusion Matrix at optimal cutoff:")
print(f"True Positives: {tp} | False Positives (Type I / Alpha): {fp}")
print(f"True Negatives: {tn} | False Negatives (Type II / Beta): {fn}")

# 3. Uplift Modeling (Conceptual simulation)
# Assume a baseline strategy has a conversion rate of 5%
# We target the top 20% of users scored by our model
top_20_percentile = np.percentile(y_prob, 80)
targeted_users = y_true[y_prob >= top_20_percentile]

model_conversion_rate = targeted_users.mean()
baseline_conversion_rate = 0.05
uplift = model_conversion_rate - baseline_conversion_rate

print(f"\nBaseline Conversion Rate: {baseline_conversion_rate*100:.1f}%")
print(f"Model Targeted Conversion Rate: {model_conversion_rate*100:.1f}%")
print(f"Absolute Uplift: {uplift*100:.1f} percentage points")
