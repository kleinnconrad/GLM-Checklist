### `1_data_preparation.py`
Demonstrates best practices for cleaning and preparing raw data for modeling.
* **Imputation:** Missing value handling using K-Nearest Neighbors (`KNNImputer`).
* **Outlier Handling:** Capping extreme values at specific percentiles (Clipping).
* **Skewness:** Applying log-transformations to highly skewed distributions.
* **Scaling:** Standardizing numerical features using `StandardScaler` (Z-transformation).
* **Binning:** Converting continuous variables into ordinal categories using quantiles (`pd.qcut`).

### `2_feature_engineering_selection.py`
Focuses on creating informative predictors and selecting the most relevant ones.
* **Pattern Features:** Transforming temporal/sequential binary data into usable numerical formats.
* **Interaction Effects:** Creating interaction terms between variables ($X_1 \times X_2$).
* **Multicollinearity:** Calculating Variance Inflation Factors (VIF) to identify and remove highly correlated features.
* **Variable Selection:** Using LASSO Regularization (`LassoCV`) to automatically penalize and drop insignificant predictors.

### `3_model_setup.py`
Illustrates how to correctly configure Generalized Linear Models based on the target variable.
* **Classification (Binomial):** Setting up a Logistic Regression equivalent using `statsmodels` with a Binomial distribution family and a Logit link function.
* **Count Data (Poisson):** Setting up a model for predicting count data using a Poisson distribution family and a Log link function.

### `4_training_validation.py`
Highlights robust validation strategies to ensure the model generalizes well to unseen data.
* **Out-of-Time Validation:** Splitting data based on strict chronological thresholds rather than random sampling to prevent data leakage.
* **Overfitting/Underfitting Checks:** Comparing Log-Loss metrics between training and validation sets to monitor model generalization.

### `5_model_evaluation_metrics.py`
Focuses on assessing model performance through a business-value lens.
* **Optimal Cutoff Point:** Determining the best classification threshold by maximizing the Kolmogorov-Smirnov (KS) statistic derived from the ROC curve.
* **Alpha & Beta Error Analysis:** Generating a Confusion Matrix at the optimal threshold to evaluate Type I (False Positives) and Type II (False Negatives) errors.
* **Uplift Simulation:** Calculating the incremental conversion rate value (Uplift) of targeting users based on the model versus a baseline strategy.

---

## Prerequisites & Installation

To run these scripts, you will need `Python 3.7+` and the following libraries:

```bash
pip install pandas numpy scikit-learn statsmodels
