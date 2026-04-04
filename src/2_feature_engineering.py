import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Generate dummy data
data = pd.DataFrame({
    'feature_A': np.random.normal(10, 2, 100),
    'feature_B': np.random.normal(5, 1, 100),
    'binary_seq': ['101', '110', '001', '111', '010'] * 20, # Temporal binary patterns
    'target': np.random.normal(50, 5, 100)
})
data['feature_C'] = data['feature_A'] * 2.5 + np.random.normal(0, 0.5, 100) # Highly correlated with A

print("\n--- 2. Feature Engineering & Selection ---")

# 1. Pattern Features (Convert binary string sequence to decimal)
data['pattern_decimal'] = data['binary_seq'].apply(lambda x: int(x, 2))

# 2. Interaction Effects (Feature A * Feature B)
data['interaction_AB'] = data['feature_A'] * data['feature_B']

# 3. Multicollinearity (VIF)
features_for_vif = data[['feature_A', 'feature_B', 'feature_C', 'interaction_AB', 'pattern_decimal']]
vif_data = pd.DataFrame()
vif_data["Feature"] = features_for_vif.columns
vif_data["VIF"] = [variance_inflation_factor(features_for_vif.values, i) for i in range(features_for_vif.shape[1])]
print("Variance Inflation Factors:\n", vif_data)

# Drop highly collinear feature C based on VIF
X = features_for_vif.drop(columns=['feature_C'])
y = data['target']

# 4. Variable Selection (LASSO Regularization for feature selection)
lasso = LassoCV(cv=5).fit(X, y)
selected_features = X.columns[lasso.coef_ != 0].tolist()
print(f"\nSelected Features via LASSO: {selected_features}")
