## Table of Contents

* [GLM Predictive Analytics Checklist](#glm-predictive-analytics-checklist)
  * [1. Data Preparation](#1-data-preparation)
  * [2. Feature Engineering & Selection](#2-feature-engineering--selection)
  * [3. Model Setup](#3-model-setup)
  * [4. Training & Validation (Validation Strategy)](#4-training--validation-validation-strategy)
  * [5. Model Evaluation & Business Metrics](#5-model-evaluation--business-metrics)
  * [Infrastructure & Orchestration: Databricks Asset Bundles (DABs)](#infrastructure--orchestration-databricks-asset-bundles-dabs)
    * [Bundle Architecture](#bundle-architecture)
  * [CI/CD Pipeline (GitHub Actions)](#-cicd-pipeline-github-actions)
    * [Workflow overview (`.github/workflows/deploy_bundle.yml`)](#️-workflow-overview-githubworkflowsdeploy_bundleyml)
    * [Required Secrets](#required-secrets)

# GLM Predictive Analytics Checklist

This checklist serves as a guide for the development, validation, and evaluation of Generalized Linear Models (GLMs) in the field of predictive analytics.

It contains an example implementation that can be deployed to Databricks with Asset Bundles.

## 1. Data Preparation
- [ ] **Imputation:** Missing values were appropriately replaced using suitable methods (e.g., mean/median imputation, KNN, model-based).
- [ ] **Outlier Handling:** Extreme data points were identified and adequately handled (e.g., via capping/clipping or transformation) to prevent distortion of the model weights.
- [ ] **Skewness:** Highly skewed distributions were analyzed and transformed if necessary (e.g., log transformation) to achieve a more symmetrical distribution.
- [ ] **Scaling & Normalization:** Numerical variables were scaled (e.g., Z-transformation / standardization) to ensure a uniform magnitude for the algorithm.
- [ ] **Categorization (Binning):** Continuous or highly granular variables were grouped into meaningful, ordinal classes if required to capture non-linear effects.

## 2. Feature Engineering & Selection
- [ ] **Pattern Features:** Complex behavioral patterns were extracted and transformed into usable formats (e.g., converting temporal binary patterns into decimal values).
- [ ] **Interaction Effects:** Relevant interactions between predictors were identified and included in the model as interaction terms (e.g., X1 * X2).
- [ ] **Multicollinearity:** High correlations between independent variables were checked (e.g., via Variance Inflation Factor - VIF), and redundant features were removed.
- [ ] **Variable Selection (Stepwise Selection):** Significant predictors were systematically selected (e.g., via forward/backward selection, LASSO regularization, or stepwise selection).

## 3. Model Setup
- [ ] **Link Function & Distribution Family:** The response variable was correctly assessed (e.g., Binomial for classification, Poisson for count data), and the appropriate link function (e.g., logit, log) was applied.

## 4. Training & Validation (Validation Strategy)
- [ ] **Out-of-Time Validation:** The data split for *Training -> Validation -> Prediction* was performed with a strict temporal shift to prevent data leakage and test real-world applicability.
- [ ] **Overfitting / Underfitting Check:** Model performance was compared across training and validation datasets to rule out excessive adaptation to training data (overfitting) or a lack of model complexity (underfitting).

## 5. Model Evaluation & Business Metrics
- [ ] **Optimal Cutoff Point:** The classification threshold was optimized in a data-driven manner based on the response profile, the ROC (Receiver Operating Characteristic) curve, or the maximum Kolmogorov-Smirnov distance (KS statistic).
- [ ] **Alpha & Beta Error Analysis:** The tolerance for False Positives (Type I error) and False Negatives (Type II error) was weighed according to the specific business case (Confusion Matrix).
- [ ] **Uplift Modeling:** The incremental value (uplift) of the model compared to a random or heuristic baseline strategy was quantified.

## Infrastructure & Orchestration: Databricks Asset Bundles (DABs)

This project leverages **Databricks Asset Bundles (DABs)** to define the entire machine learning pipeline as code (Infrastructure as Code). This ensures reproducibility, version control, and seamless deployment across environments.

### Bundle Architecture
The configuration is modularized to separate workspace settings from the actual job logic:

* **`databricks.yml`:** The central entry point. It defines the target Databricks workspace and instructs the bundle to include all resource definitions from the `resources/` directory.
* **`resources/glm_pipeline.yml`:** The job definition. It configures a multi-task Databricks Job running on **Serverless Compute**. 
  * **Task Dependencies:** The job strictly follows the "Read-Transform-Write" pattern. Tasks are linked using `depends_on` (e.g., Feature Engineering only starts after Data Preparation successfully finishes).
  * **Serverless Notebook Tasks:** To efficiently utilize Serverless Compute and bypass complex environment dependency definitions, the Python scripts in `src/` are executed as `notebook_task`. The `# Databricks notebook source` magic comment in the scripts allows Databricks to run them natively using the pre-configured Databricks ML runtime (which includes `scikit-learn`, `pandas`, `statsmodels`, etc.).

---

## CI/CD Pipeline (GitHub Actions)

To ensure that the Databricks workspace is always in sync with the codebase, this repository includes a Continuous Integration and Continuous Deployment (CI/CD) pipeline powered by **GitHub Actions**.

### Workflow overview (`.github/workflows/deploy_bundle.yml`)

The pipeline automatically triggers on any `push` to the `main` branch, provided the changes involve the Python scripts (`src/`), the bundle configurations (`resources/`), or the `databricks.yml` file.

1. **Checkout Code:** Retrieves the latest commit from the repository.
2. **Setup Databricks CLI:** Installs the official Databricks CLI on the runner.
3. **Validate Bundle:** Runs `databricks bundle validate` to strictly check the YAML syntax and cluster configurations. This catches configuration errors *before* touching the live Databricks environment.
4. **Deploy Bundle:** Runs `databricks bundle deploy`, which uses Terraform under the hood to update the Databricks Job, upload the latest Python scripts, and apply any structural changes to the pipeline.

### Required Secrets
To run this pipeline in your own fork/environment, the following GitHub Repository Secrets must be configured under *Settings > Secrets and variables > Actions*:

* `DATABRICKS_HOST`: Your Databricks workspace URL (e.g., `https://dbc-xxxx.cloud.databricks.com`).
* `DATABRICKS_TOKEN`: A valid Databricks Personal Access Token (PAT) with permissions to create and run jobs in the target workspace.
