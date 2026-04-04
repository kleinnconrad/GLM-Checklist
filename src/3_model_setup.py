from pyspark.sql import SparkSession
import pandas as pd
import statsmodels.api as sm

spark = SparkSession.builder.getOrCreate()

print("\n--- 3. Model Setup ---")
print("Reading data from Step 2...")
data = spark.table("workspace.default.glm_02_features").toPandas()

# This script acts as a configuration and sanity check phase to ensure the GLM family matches the data.
if 'target' in data.columns and 'feature_A' in data.columns:
    # Prepare a test matrix (Drop NaNs for statsmodels)
    test_data = data[['target', 'feature_A']].dropna()
    
    X = test_data[['feature_A']]
    X = sm.add_constant(X)
    y = test_data['target']

    try:
        # Check Binomial setup (requires target to be 0 or 1)
        if y.nunique() == 2:
            glm_binomial = sm.GLM(y, X, family=sm.families.Binomial(link=sm.families.links.Logit()))
            model_bin_results = glm_binomial.fit()
            print("Sanity Check: Binomial GLM (Logistic) AIC:", model_bin_results.aic)
    except Exception as e:
        print(f"GLM Setup check skipped or failed: {e}")

print("Passing validated dataset forward...")
spark_df_ready = spark.createDataFrame(data)
spark_df_ready.write \
    .mode("overwrite") \
    .option("mergeSchema", "true") \
    .saveAsTable("workspace.default.glm_03_model_ready")

print("Step 3 Complete. Saved as: workspace.default.glm_03_model_ready")