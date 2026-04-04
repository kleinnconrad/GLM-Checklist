# Databricks notebook source
# MAGIC %pip install statsmodels

# COMMAND ----------

from pyspark.sql import SparkSession
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

spark = SparkSession.builder.getOrCreate()

print("--- 1. Data Preparation ---")
print("Reading raw data from workspace.default.test_population...")
df_spark = spark.table("workspace.default.test_population")
data = df_spark.toPandas()

# 1. Imputation (KNN)
# Replace 'age' with your actual column name containing missing values
if 'age' in data.columns:
    imputer = KNNImputer(n_neighbors=3)
    data['age_imputed'] = imputer.fit_transform(data[['age']])

# 2. Outlier Handling
if 'score' in data.columns:
    lower_bound = data['score'].quantile(0.01)
    upper_bound = data['score'].quantile(0.99)
    data['score_clipped'] = np.clip(data['score'], lower_bound, upper_bound)

# 3. Skewness
if 'income' in data.columns:
    data['income_log'] = np.log1p(data['income'])

# 4. Scaling
if 'age_imputed' in data.columns:
    scaler = StandardScaler()
    data['age_scaled'] = scaler.fit_transform(data[['age_imputed']])

# 5. Binning
if 'age_imputed' in data.columns:
    # We cast to string immediately because Spark does not support Pandas categorical types
    data['age_binned'] = pd.qcut(data['age_imputed'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4']).astype(str)

print("Saving prepared data for the next step...")
# Convert back to Spark DataFrame and write to Unity Catalog
spark_df_prepared = spark.createDataFrame(data)
spark_df_prepared.write \
    .mode("overwrite") \
    .option("mergeSchema", "true") \
    .saveAsTable("workspace.default.glm_01_prepared")

print("Step 1 Complete. Saved as: workspace.default.glm_01_prepared")