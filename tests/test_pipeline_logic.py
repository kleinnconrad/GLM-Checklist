import pytest
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve

# ==========================================
# 1. PIPELINE LOGIC (Ideally imported from src/utils.py)
# ==========================================

def convert_binary_seq_to_decimal(seq):
    """Converts a binary sequence string to a decimal integer."""
    if pd.isnull(seq) or seq == "":
        return 0
    return int(str(seq), 2)

def clip_outliers(series, lower_quantile=0.01, upper_quantile=0.99):
    """Clips extreme values in a Pandas Series based on quantiles."""
    lower_bound = series.quantile(lower_quantile)
    upper_bound = series.quantile(upper_quantile)
    return np.clip(series, lower_bound, upper_bound)

def calculate_optimal_cutoff(y_true, y_prob):
    """Calculates the optimal classification threshold using the KS statistic."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    ks_stat = tpr - fpr
    optimal_idx = np.argmax(ks_stat)
    return thresholds[optimal_idx]

def calculate_uplift(y_true, y_prob, target_percentile=80):
    """Calculates the absolute uplift of targeting top percentiles vs baseline."""
    baseline_conversion = np.mean(y_true)
    
    threshold = np.percentile(y_prob, target_percentile)
    targeted_users = [y for y, p in zip(y_true, y_prob) if p >= threshold]
    
    if len(targeted_users) == 0:
        return 0.0
        
    model_conversion = np.mean(targeted_users)
    return model_conversion - baseline_conversion


# ==========================================
# 2. UNIT TESTS (Run via pytest)
# ==========================================

def test_convert_binary_seq_to_decimal():
    # Test normal binary strings
    assert convert_binary_seq_to_decimal("101") == 5
    assert convert_binary_seq_to_decimal("1111") == 15
    assert convert_binary_seq_to_decimal("000") == 0
    
    # Test edge cases
    assert convert_binary_seq_to_decimal(None) == 0
    assert convert_binary_seq_to_decimal("") == 0

def test_clip_outliers():
    # Create a deterministic series from 1 to 100
    data = pd.Series(range(1, 101))
    
    # Clip at the 10th and 90th percentiles (10.9 and 90.1)
    clipped_data = clip_outliers(data, lower_quantile=0.10, upper_quantile=0.90)
    
    assert round(clipped_data.min(), 1) == 10.9
    assert round(clipped_data.max(), 1) == 90.1
    # Check that the length of the series hasn't changed
    assert len(clipped_data) == 100

def test_calculate_optimal_cutoff():
    # Create perfectly separable dummy data
    y_true = [0, 0, 0, 1, 1, 1]
    y_prob = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    
    # The optimal cutoff should clearly separate the 0s and 1s (e.g., >= 0.7)
    optimal_threshold = calculate_optimal_cutoff(y_true, y_prob)
    
    assert optimal_threshold >= 0.3
    assert optimal_threshold <= 0.7

def test_calculate_uplift():
    # Dummy data where top probabilities perfectly match the targets
    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.9, 0.95])
    
    # Baseline conversion is 2/6 = 0.333...
    # If we target the top 33.3% (percentile 66.6), we capture only the 1s.
    # Model conversion for targeted group = 2/2 = 1.0
    # Uplift = 1.0 - 0.333... = 0.666...
    
    uplift = calculate_uplift(y_true, y_prob, target_percentile=66.6)
    
    assert round(uplift, 2) == 0.67

def test_calculate_uplift_empty_target():
    y_true = [0, 0, 1]
    y_prob = [0.1, 0.2, 0.3]
    # Targeting a percentile so high nobody qualifies
    uplift = calculate_uplift(y_true, y_prob, target_percentile=100)
    assert uplift == 0.0
  
