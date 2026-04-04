import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

# Generate dummy data
np.random.seed(42)
data = pd.DataFrame({
    'age': np.random.normal(40, 15, 100),
    'income': np.random.exponential(50000, 100), # Highly skewed
    'score': np.random.normal(50, 10, 100)
})
# Introduce missing values and outliers
data.loc[10:15, 'age'] = np.nan
data.loc[0, 'score'] = 9999 

print("--- 1. Data Preparation ---")

# 1. Imputation (KNN)
imputer = KNNImputer(n_neighbors=3)
data['age_imputed'] = imputer.fit_transform(data[['age']])

# 2. Outlier Handling (Clipping at 1st and 99th percentiles)
lower_bound = data['score'].quantile(0.01)
upper_bound = data['score'].quantile(0.99)
data['score_clipped'] = np.clip(data['score'], lower_bound, upper_bound)

# 3. Skewness (Log-transformation for income)
data['income_log'] = np.log1p(data['income'])

# 4. Scaling & Normalization (Z-Transformation)
scaler = StandardScaler()
data['age_scaled'] = scaler.fit_transform(data[['age_imputed']])

# 5. Categorization / Binning (Ordinal classes for age)
data['age_binned'] = pd.qcut(data['age_imputed'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

print(data[['age_imputed', 'score_clipped', 'income_log', 'age_scaled', 'age_binned']].head())
