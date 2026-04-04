# Databricks notebook source
from pyspark.sql import SparkSession
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve

spark = SparkSession.builder.getOrCreate()

print("\n--- 5. Model Evaluation & Business Metrics ---")
print("Reading predictions from Step 4...")
data = spark.table("workspace.default.glm_04_predictions").toPandas()

if 'target' in data.columns and 'predicted_probability' in data.columns:
    y_true = data['target']
    y_prob = data['predicted_probability']

    # 1. Optimal Cutoff Point (via KS Statistic / ROC Curve)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    ks_stat = tpr - fpr
    optimal_idx = np.argmax(ks_stat)
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal Cutoff Threshold (Max KS): {optimal_threshold:.4f}")

    # 2. Alpha & Beta Error Analysis (Confusion Matrix)
    y_pred_optimal = (y_prob >= optimal_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_optimal).ravel()
    print(f"\nConfusion Matrix at optimal cutoff ({optimal_threshold:.4f}):")
    print(f"True Positives: {tp} | False Positives (Type I / Alpha): {fp}")
    print(f"True Negatives: {tn} | False Negatives (Type II / Beta): {fn}")

    # 3. Uplift Modeling (Conceptual simulation)
    # We target the top 20% of users scored by our model
    top_20_percentile = np.percentile(y_prob, 80)
    targeted_users = y_true[y_prob >= top_20_percentile]

    if len(targeted_users) > 0:
        model_conversion_rate = targeted_users.mean()
        # Assume a baseline strategy has a global conversion rate
        baseline_conversion_rate = y_true.mean() 
        uplift = model_conversion_rate - baseline_conversion_rate

        print(f"\nBaseline Conversion Rate (Global): {baseline_conversion_rate*100:.1f}%")
        print(f"Model Targeted Conversion Rate (Top 20%): {model_conversion_rate*100:.1f}%")
        print(f"Absolute Uplift: {uplift*100:.1f} percentage points")
    else:
        print("Not enough variance in predictions to calculate uplift.")

print("\n✅ Pipeline Execution Finished.")