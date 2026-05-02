# ============================================
# Cell 1: Install and Import Libraries
# ============================================

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import logging
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler  # Changed from StandardScaler
from sklearn.svm import OneClassSVM  # For comparison
from sklearn.metrics import (precision_score, recall_score, f1_score,
                            confusion_matrix, classification_report,
                            roc_auc_score, average_precision_score,
                            precision_recall_curve, roc_curve, balanced_accuracy_score)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ============================================
# Cell 2: Data Loading Functions
# ============================================

def load_unsw_data(raw_dir='../data/raw'):
    """Load UNSW-NB15 CSV files with encoding handling"""
    raw_path = Path(raw_dir)

    if not raw_path.exists():
        raise FileNotFoundError(f"Directory not found: {raw_path}")

    csv_files = list(raw_path.glob('*.csv'))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {raw_path}")

    logger.info(f"Found {len(csv_files)} CSV files")

    dfs = []
    for file in csv_files:
        logger.info(f"Loading {file.name}...")

        # Try different encodings
        df = None
        for encoding in ['latin1', 'utf-8', 'iso-8859-1']:
            try:
                df = pd.read_csv(file, low_memory=False, encoding=encoding)
                logger.info(f"  Loaded with {encoding} encoding")
                break
            except:
                continue

        if df is None:
            raise ValueError(f"Could not load {file.name}")

        dfs.append(df)
        logger.info(f"  Loaded {len(df):,} rows")

    combined_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total rows loaded: {len(combined_df):,}")

    return combined_df




