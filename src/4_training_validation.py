# Databricks notebook source
from pyspark.sql import SparkSession
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

spark = SparkSession.builder.getOrCreate()

print("\n--- 4. Training & Validation ---")
print("Reading data from Step 3...")
data = spark.table("workspace.default.glm_03_model_ready").toPandas()

# Ensure date column is datetime format for out-of-time splitting
if 'date' in data.columns and 'target' in data.columns and 'feature_A' in data.columns:
    data['date'] = pd.to_datetime(data['date'])

    # 1. Out-of-Time Validation (Adjust dates to your actual dataset timeline!)
    split_date = '2025-01-01'
    train_data = data[data['date'] < split_date].copy()
    val_data = data[data['date'] >= split_date].copy()

    # Define features to use
    features = ['feature_A'] # Add more features here based on step 2
    
    X_train, y_train = train_data[features].fillna(0), train_data['target']
    X_val, y_val = val_data[features].fillna(0), val_data['target']

    # 2. Train Model
    print("Training Logistic Regression Model...")
    model = LogisticRegression(max_iter=1000).fit(X_train, y_train)

    # 3. Generate Predictions
    train_preds = model.predict_proba(X_train)[:, 1]
    val_preds = model.predict_proba(X_val)[:, 1]

    # Overfitting Check
    train_loss = log_loss(y_train, train_preds)
    val_loss = log_loss(y_val, val_preds)
    print(f"Training Log-Loss: {train_loss:.4f} | Validation Log-Loss: {val_loss:.4f}")

    # Append predictions back to the validation dataframe for Step 5
    val_data['predicted_probability'] = val_preds
    
    print("Saving validation set with predictions...")
    # Convert validation data with predictions to Spark and save
    spark_df_preds = spark.createDataFrame(val_data)
    spark_df_preds.write \
        .mode("overwrite") \
        .option("mergeSchema", "true") \
        .saveAsTable("workspace.default.glm_04_predictions")

    print("Step 4 Complete. Saved as: workspace.default.glm_04_predictions")
else:
    print("Missing 'date', 'target', or 'feature_A' columns. Cannot execute split.")