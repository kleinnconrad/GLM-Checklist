from pyspark.sql import SparkSession
import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from statsmodels.stats.outliers_influence import variance_inflation_factor

spark = SparkSession.builder.getOrCreate()

print("\n--- 2. Feature Engineering & Selection ---")
print("Reading data from Step 1...")
data = spark.table("workspace.default.glm_01_prepared").toPandas()

# 1. Pattern Features
if 'binary_seq' in data.columns:
    data['pattern_decimal'] = data['binary_seq'].apply(lambda x: int(str(x), 2) if pd.notnull(x) else 0)

# 2. Interaction Effects
if 'feature_A' in data.columns and 'feature_B' in data.columns:
    data['interaction_AB'] = data['feature_A'] * data['feature_B']

# 3. Multicollinearity (VIF) - Example with selected numeric columns
numeric_cols = [col for col in ['feature_A', 'feature_B', 'interaction_AB', 'pattern_decimal'] if col in data.columns]
if len(numeric_cols) > 1:
    features_for_vif = data[numeric_cols].fillna(0) # VIF cannot handle NaNs
    vif_data = pd.DataFrame()
    vif_data["Feature"] = features_for_vif.columns
    vif_data["VIF"] = [variance_inflation_factor(features_for_vif.values, i) for i in range(features_for_vif.shape[1])]
    print("Variance Inflation Factors:\n", vif_data)

# Pass the enriched dataset forward
print("Saving engineered features for the next step...")
spark_df_features = spark.createDataFrame(data)
spark_df_features.write \
    .mode("overwrite") \
    .option("mergeSchema", "true") \
    .saveAsTable("workspace.default.glm_02_features")

print("Step 2 Complete. Saved as: workspace.default.glm_02_features")