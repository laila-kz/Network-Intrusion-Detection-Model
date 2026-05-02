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


class IsolationForestTrainer_Optimized:
    def __init__(self, contamination=0.01, random_state=42):  # Explicit contamination
        self.contamination = contamination
        self.random_state = random_state
        self.model = None
        self.threshold = None
        
    def train(self, X_normal, sample_size=None):
        """Train ONLY on normal samples with optimized hyperparameters"""
        logger.info("\n" + "="*60)
        logger.info("TRAINING ISOLATION FOREST (Optimized)")
        logger.info("="*60)
        
        # Verify we only have normal samples
        logger.info(f"Training on {len(X_normal):,} NORMAL samples only")
        
        # Sample if needed (max 50k for efficiency)
        if sample_size is None:
            sample_size = min(50000, len(X_normal))
        
        if len(X_normal) > sample_size:
            indices = np.random.choice(len(X_normal), sample_size, replace=False)
            X_normal = X_normal[indices]
            logger.info(f"Sampled {sample_size:,} normal samples for training")
        
        # Initialize model with OPTIMIZED hyperparameters
        self.model = IsolationForest(
            contamination=self.contamination,  # Explicit: 1% expected anomalies
            random_state=self.random_state,
            n_estimators=500,      # Increased for stability
            max_samples=0.8,       # Use 80% of samples per tree
            bootstrap=True,        # Bootstrap sampling
            max_features=0.7,      # Feature randomness to prevent overfitting
            verbose=1
        )
        
        self.model.fit(X_normal)
        logger.info("✅ Training completed!")
        return self.model
    
    def predict_with_threshold(self, X, threshold_percentile=10):
        """Predict with custom threshold based on anomaly scores"""
        scores = self.model.decision_function(X)
        threshold = np.percentile(scores, threshold_percentile)
        predictions = np.where(scores < threshold, -1, 1)
        return predictions, scores, threshold
    
    def predict(self, X):
        """Standard prediction"""
        return self.model.predict(X)
    
    def find_optimal_f1_threshold(self, X_val, y_val):
        """Find threshold that maximizes F1 Score (not accuracy)"""
        scores = self.model.decision_function(X_val)
        
        # Try all possible thresholds based on score distribution
        thresholds = np.linspace(scores.min(), scores.max(), 200)
        
        best_f1 = 0
        best_threshold = 0
        best_predictions = None
        best_precision = 0
        best_recall = 0
        
        f1_scores = []
        
        for threshold in thresholds:
            pred_binary = (scores >= threshold).astype(int)  # 1=normal, 0=attack
            
            # Skip if no predictions in one class
            if len(np.unique(pred_binary)) < 2:
                continue
                
            f1 = f1_score(y_val, pred_binary, zero_division=0)
            f1_scores.append((threshold, f1))
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_predictions = pred_binary
                best_precision = precision_score(y_val, pred_binary, zero_division=0)
                best_recall = recall_score(y_val, pred_binary, zero_division=0)
        
        logger.info(f"Optimal threshold: {best_threshold:.4f} (F1={best_f1:.4f}, P={best_precision:.4f}, R={best_recall:.4f})")
        self.threshold = best_threshold
        
        return best_threshold, best_predictions, f1_scores

# Split data
print("\n" + "="*70)
print("SPLITTING DATA")
print("="*70)

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)

# Train ONLY on normal samples
X_normal_train = X_train[y_train == 1]
print(f"\nTraining data:")
print(f"  Normal samples for training: {len(X_normal_train):,}")
print(f"  Attack samples EXCLUDED from training: {(y_train==0).sum():,}")

# Train optimized model
trainer = IsolationForestTrainer_Optimized(contamination=0.01, random_state=42)
model = trainer.train(X_normal_train)

print(f"\nModel Configuration:")
print(f"  n_estimators: {model.n_estimators}")
print(f"  max_features: {model.max_features}")
print(f"  contamination: {model.contamination}")





#=============================
# Evaluate on validation set to find optimal threshold
# ============================================


# Get scores for all datasets
train_scores = model.decision_function(X_train)
val_scores = model.decision_function(X_val)
test_scores = model.decision_function(X_test)

# Visualize score distributions to identify overlap
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram of scores by class (Validation set)
ax1 = axes[0]
ax1.hist(val_scores[y_val == 1], bins=50, alpha=0.5, label='Normal (train)', color='green', density=True)
ax1.hist(val_scores[y_val == 0], bins=50, alpha=0.5, label='Attack (anomaly)', color='red', density=True)
ax1.axvline(x=0, color='black', linestyle='--', label='Default threshold (0)')
if hasattr(trainer, 'threshold') and trainer.threshold:
    ax1.axvline(x=trainer.threshold, color='blue', linestyle='--', label=f'Optimal F1 threshold ({trainer.threshold:.3f})')
ax1.set_xlabel('Anomaly Score (higher = more normal)')
ax1.set_ylabel('Density')
ax1.set_title('Score Distribution by Class (Validation)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Zoomed view to see overlap region
ax2 = axes[1]
# Get score range around zero for better visibility
score_min = min(-0.3, val_scores.min())
score_max = max(0.3, val_scores.max())
ax2.hist(val_scores[(y_val == 1) & (val_scores > score_min) & (val_scores < score_max)], 
         bins=50, alpha=0.5, label='Normal', color='green', density=True)
ax2.hist(val_scores[(y_val == 0) & (val_scores > score_min) & (val_scores < score_max)], 
         bins=50, alpha=0.5, label='Attack', color='red', density=True)
ax2.axvline(x=0, color='black', linestyle='--', label='Default threshold')
if hasattr(trainer, 'threshold') and trainer.threshold:
    ax2.axvline(x=trainer.threshold, color='blue', linestyle='--', label=f'Optimal threshold')
ax2.set_xlabel('Anomaly Score (higher = more normal)')
ax2.set_ylabel('Density')
ax2.set_title('Score Distribution (Zoomed - Overlap Region)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Calculate overlap percentage
print("\n" + "="*70)
print("SCORE DISTRIBUTION ANALYSIS")
print("="*70)

# Define overlap region (where both classes have density > 0)
overlap_threshold = 0  # Scores around zero
normal_overlap = ((val_scores[y_val == 1] > -0.1) & (val_scores[y_val == 1] < 0.1)).mean()
attack_overlap = ((val_scores[y_val == 0] > -0.1) & (val_scores[y_val == 0] < 0.1)).mean()
print(f"Percentage of samples in overlap zone (-0.1 to 0.1):")
print(f"  Normal samples: {normal_overlap:.1%}")
print(f"  Attack samples: {attack_overlap:.1%}")