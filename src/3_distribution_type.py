import pandas as pd
import numpy as np
import statsmodels.api as sm

# Generate dummy data
np.random.seed(42)
X = pd.DataFrame({'feature_1': np.random.normal(0, 1, 100), 'feature_2': np.random.normal(0, 1, 100)})
X = sm.add_constant(X)

# Binary target for Logistic Regression
y_binary = np.random.binomial(1, p=1/(1+np.exp(-X['feature_1'])), size=100)

# Count target for Poisson Regression
y_count = np.random.poisson(lam=np.exp(X['feature_1']), size=100)

print("\n--- 3. Model Setup ---")

# 1. Classification: Binomial distribution with Logit link
glm_binomial = sm.GLM(y_binary, X, family=sm.families.Binomial(link=sm.families.links.Logit()))
model_bin_results = glm_binomial.fit()
print("Binomial GLM (Logistic) AIC:", model_bin_results.aic)

# 2. Count Data: Poisson distribution with Log link
glm_poisson = sm.GLM(y_count, X, family=sm.families.Poisson(link=sm.families.links.Log()))
model_pois_results = glm_poisson.fit()
print("Poisson GLM AIC:", model_pois_results.aic)
