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
# Setup logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class UNSWPreprocessor_Optimized:
    def __init__(self):
        self.scaler = RobustScaler()  # Changed to RobustScaler for skewed data
        self.feature_columns = None
        self.correlation_threshold = 0.95
        
        # Define columns to ALWAYS drop (unique identifiers, no predictive value)
        self.always_drop = [
            'id', 'ID', 'flow_id', 'Flow ID', 'srcip', 'dstip',
            'source_ip', 'destination_ip', 'timestamp', 'time',
            'sport', 'dsport',  # Ports often cause overfitting
            'stos', 'dtos', 'sttl', 'dttl',  # TTL values vary by OS, not attack signature
        ]
        
        # High cardinality categorical features to drop (or encode separately)
        self.drop_categorical = ['proto', 'service', 'state']

    def find_label_column(self, df):
        """Find label column in UNSW-NB15 dataset"""
        possible_labels = ['label', 'attack_cat', 'Label', 'ATTACK', 'class']
        
        for col in possible_labels:
            if col in df.columns:
                logger.info(f"Found label column: '{col}'")
                return col
        
        for col in df.columns:
            if 'label' in col.lower() or 'attack' in col.lower():
                logger.info(f"Found potential label column: '{col}'")
                return col
        
        logger.info(f"Available columns: {list(df.columns[:20])}")
        raise ValueError("No label column found in dataset")
    
    def remove_high_correlation_features(self, X):
        """Remove highly correlated features to reduce noise"""
        # Compute correlation matrix
        corr_matrix = X.corr().abs()
        
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features with correlation > threshold
        high_corr_features = [column for column in upper.columns if any(upper[column] > self.correlation_threshold)]
        
        if high_corr_features:
            logger.info(f"Removing {len(high_corr_features)} highly correlated features (corr > {self.correlation_threshold})")
            X = X.drop(columns=high_corr_features)
        
        return X, high_corr_features

    def preprocess(self, df, label_col):
        """Complete preprocessing with improved feature engineering"""
        logger.info("\n" + "="*60)
        logger.info("STARTING DATA PREPROCESSING (Optimized)")
        logger.info("="*60)
        
        # Extract labels
        y = df[label_col].copy()
        
        # Check label values
        logger.info(f"Unique label values: {y.unique()[:10]}")
        
        # IMPORTANT: UNSW-NB15 uses 0=normal, 1=attack in numeric labels
        # We want: 1=normal (BENIGN), 0=attack for our model
        if y.dtype == 'object':
            y = y.astype(str).str.lower().str.strip()
            # 'normal' = normal traffic, everything else = attack
            y_binary = (y == 'normal').astype(int)
            logger.info("Labels mapped: 'normal' -> 1, else -> 0")
        else:
            # Numeric labels: assume 0=normal, 1=attack
            y_binary = (y == 0).astype(int)
            logger.info("Labels mapped: 0 -> 1 (normal), 1 -> 0 (attack)")
        
        # Remove label column
        X = df.drop(columns=[label_col])
        
        # Remove attack_cat if exists
        if 'attack_cat' in X.columns:
            X = X.drop(columns=['attack_cat'])
            logger.info("Removed 'attack_cat' column")
        
        # STEP 1: Remove always-drop columns (IDs, IPs, timestamps)
        cols_to_remove = [col for col in self.always_drop if col in X.columns]
        if cols_to_remove:
            X = X.drop(columns=cols_to_remove)
            logger.info(f"Removed ID/identifier columns: {cols_to_remove}")
        
        # STEP 2: Remove high-cardinality categorical columns
        cols_to_remove = [col for col in self.drop_categorical if col in X.columns]
        if cols_to_remove:
            X = X.drop(columns=cols_to_remove)
            logger.info(f"Removed high-cardinality categorical columns: {cols_to_remove}")
        
        logger.info(f"Shape after identifier/categorical removal: {X.shape}")
        
        # STEP 3: Keep only numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
        
        if len(non_numeric_cols) > 0:
            logger.info(f"Dropping {len(non_numeric_cols)} remaining non-numeric columns")
            logger.info(f"  Columns: {list(non_numeric_cols)}")
            X = X[numeric_cols]
        
        # STEP 4: Handle missing and infinite values
        if X.isnull().any().any():
            X = X.fillna(0)
            logger.info("Filled missing values with 0")
        
        X = X.replace([np.inf, -np.inf], 0)
        
        # STEP 5: Remove constant columns
        constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
        if constant_cols:
            X = X.drop(columns=constant_cols)
            logger.info(f"Dropped {len(constant_cols)} constant columns")
        
        # STEP 6: Remove highly correlated features (BEFORE scaling)
        X, high_corr_features = self.remove_high_correlation_features(X)
        
        # STEP 7: Log transform for skewed features (optional - RobustScaler handles this well)
        # Only apply to features with extreme skew > 5
        for col in X.columns:
            if X[col].skew() > 5:
                X[col] = np.log1p(X[col].clip(lower=0))
                # logger.info(f"Applied log transform to {col}")  # Too verbose, uncomment if needed
        
        # STEP 8: Store feature names and scale
        self.feature_columns = X.columns.tolist()
        logger.info(f"Final feature count: {len(self.feature_columns)}")
        
        # Apply RobustScaler (better for network traffic data with outliers)
        X_scaled = self.scaler.fit_transform(X)
        
        # Log distribution
        logger.info(f"\n📊 Final Label Distribution:")
        normal_count = (y_binary == 1).sum()
        attack_count = (y_binary == 0).sum()
        logger.info(f"  Normal (1): {normal_count:,} ({normal_count/len(y_binary)*100:.2f}%)")
        logger.info(f"  Attack (0): {attack_count:,} ({attack_count/len(y_binary)*100:.2f}%)")
        
        return X_scaled, y_binary, X  # Return original X for correlation analysis
    
    def save_scaler(self, path='models/scaler_robust.pkl'):
        full_path = PROJECT_ROOT / path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, full_path)
        logger.info(f"✅ Scaler saved to {full_path}")

# Load and preprocess
print("\n" + "="*70)
print("LOADING AND PREPROCESSING DATA (Optimized)")
print("="*70)

df = load_unsw_data('data/raw')
preprocessor = UNSWPreprocessor_Optimized()
label_col = preprocessor.find_label_column(df)
X, y, X_original = preprocessor.preprocess(df, label_col)
preprocessor.save_scaler()

print(f"\n✅ Preprocessing complete!")
print(f"Features shape: {X.shape}")
print(f"Normal samples (1): {(y==1).sum():,}")
print(f"Attack samples (0): {(y==0).sum():,}")