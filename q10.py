"""
Q10: Classification model predicting whether a professor receives a "pepper"
from all available factors (tags + numerical predictors).

Includes AUC-ROC and addresses class imbalance concerns.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


def train_test_split_indices_stratified(y: np.ndarray, test_size: float = 0.2, seed: int = 0):
    """
    Stratified train/test split to maintain class proportions in both sets.
    """
    rng = np.random.default_rng(seed)
    n = len(y)

    # Split indices by class
    idx_0 = np.where(y == 0)[0]
    idx_1 = np.where(y == 1)[0]

    # Shuffle each class
    rng.shuffle(idx_0)
    rng.shuffle(idx_1)

    # Split each class proportionally
    n_test_0 = int(round(len(idx_0) * test_size))
    n_test_1 = int(round(len(idx_1) * test_size))

    test_idx = np.concatenate([idx_0[:n_test_0], idx_1[:n_test_1]])
    train_idx = np.concatenate([idx_0[n_test_0:], idx_1[n_test_1:]])

    return train_idx, test_idx


def standardize_train_test(X_train: np.ndarray, X_test: np.ndarray):
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0, ddof=0)
    sigma = np.where(sigma < 1e-12, 1.0, sigma)
    return (X_train - mu) / sigma, (X_test - mu) / sigma, mu, sigma


def add_intercept(X: np.ndarray):
    return np.column_stack([np.ones(X.shape[0]), X])


def sigmoid(z):
    # Clip to avoid overflow
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def logistic_regression_fit(X: np.ndarray, y: np.ndarray, class_weights: dict = None,
                            lr: float = 0.1, max_iter: int = 1000, tol: float = 1e-6):
    """
    Logistic regression via gradient descent with optional class weights.
    X should include intercept column.
    """
    n, p = X.shape
    beta = np.zeros(p)

    # Compute sample weights based on class weights
    if class_weights is not None:
        sample_weights = np.array([class_weights[int(yi)] for yi in y])
    else:
        sample_weights = np.ones(n)

    for iteration in range(max_iter):
        z = X @ beta
        prob = sigmoid(z)

        # Weighted gradient
        error = (prob - y) * sample_weights
        gradient = X.T @ error / n

        # Update
        beta_new = beta - lr * gradient

        # Check convergence
        if np.max(np.abs(beta_new - beta)) < tol:
            break
        beta = beta_new

    return beta


def predict_proba(X: np.ndarray, beta: np.ndarray):
    return sigmoid(X @ beta)


def predict_class(X: np.ndarray, beta: np.ndarray, threshold: float = 0.5):
    return (predict_proba(X, beta) >= threshold).astype(int)


def roc_auc_score(y_true: np.ndarray, y_prob: np.ndarray):
    """
    Compute AUC-ROC using the Mann-Whitney U statistic formulation.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    n_pos = len(pos_idx)
    n_neg = len(neg_idx)

    if n_pos == 0 or n_neg == 0:
        return np.nan

    # Count pairs where positive has higher probability than negative
    pos_probs = y_prob[pos_idx]
    neg_probs = y_prob[neg_idx]

    # Efficient calculation using sorting
    count = 0
    for p in pos_probs:
        count += np.sum(p > neg_probs) + 0.5 * np.sum(p == neg_probs)

    auc = count / (n_pos * n_neg)
    return auc


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray):
    """Returns TN, FP, FN, TP"""
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    return tn, fp, fn, tp


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray):
    """Compute various classification metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred)

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    auc = roc_auc_score(y_true, y_prob)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "auc_roc": auc,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp
    }


def compute_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, n_thresholds: int = 100):
    """
    Compute ROC curve points (FPR, TPR) at various thresholds.
    """
    thresholds = np.linspace(0, 1, n_thresholds)
    tpr_list = []
    fpr_list = []

    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred)

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    return np.array(fpr_list), np.array(tpr_list), thresholds


# -----------------------
# Load + label columns
# -----------------------
num = pd.read_csv("rmpCapstoneNum.csv", header=None)
num.columns = [
    "avg_rating",
    "avg_difficulty",
    "num_ratings",
    "pepper",
    "would_take_again",
    "num_online",
    "male",
    "female",
]

tags = pd.read_csv("rmpCapstoneTags.csv", header=None)
tags.columns = [
    "tough_grader",
    "good_feedback",
    "respected",
    "lots_to_read",
    "participation_matters",
    "dont_skip",
    "lots_homework",
    "inspirational",
    "pop_quizzes",
    "accessible",
    "many_papers",
    "clear_grading",
    "hilarious",
    "test_heavy",
    "graded_few_things",
    "amazing_lectures",
    "caring",
    "extra_credit",
    "group_projects",
    "lecture_heavy",
]

# Combine dataframes
df = pd.concat([num, tags], axis=1)

# -----------------------
# Filter: clear gender + 10+ ratings (consistent with previous questions)
# -----------------------
df = df.loc[(df["male"] + df["female"] == 1)].copy()
df = df.loc[df["num_ratings"] >= 10].copy()
df["male_prof"] = df["male"]

print("After filtering (clear gender + 10+ ratings):")
print("Total:", len(df))
print("-" * 70)

# -----------------------
# Build design matrix - predicting pepper
# -----------------------
y = df["pepper"]

# Predictors: numerical (excluding pepper and redundant gender) + all tags
numeric_features = ["avg_rating", "avg_difficulty", "num_ratings", "would_take_again", "num_online", "male_prof"]
tag_features = list(tags.columns)
feature_cols = numeric_features + tag_features

X = df[feature_cols]

# Drop rows with missing values
model_df = pd.concat([y, X], axis=1).dropna().copy()
y = model_df["pepper"].to_numpy(dtype=int)
X = model_df[feature_cols].to_numpy(dtype=float)

print(f"Rows used in classification after dropping missing values: {len(y)}")
print("-" * 70)

# -----------------------
# Check class imbalance
# -----------------------
n_pepper = np.sum(y == 1)
n_no_pepper = np.sum(y == 0)
print("CLASS DISTRIBUTION:")
print(f"  No pepper (0): {n_no_pepper} ({100*n_no_pepper/len(y):.1f}%)")
print(f"  Pepper (1):    {n_pepper} ({100*n_pepper/len(y):.1f}%)")
print(f"  Imbalance ratio: {n_no_pepper/n_pepper:.2f}:1")
print("-" * 70)

# -----------------------
# Stratified train/test split
# -----------------------
train_idx, test_idx = train_test_split_indices_stratified(y, test_size=0.2, seed=0)
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print(f"Train set: {len(y_train)} samples (pepper: {np.sum(y_train)}, no pepper: {len(y_train)-np.sum(y_train)})")
print(f"Test set:  {len(y_test)} samples (pepper: {np.sum(y_test)}, no pepper: {len(y_test)-np.sum(y_test)})")
print("-" * 70)

# Standardize features
X_train_z, X_test_z, mu, sigma = standardize_train_test(X_train, X_test)
X_train_i = add_intercept(X_train_z)
X_test_i = add_intercept(X_test_z)

# -----------------------
# Address class imbalance with class weights
# -----------------------
# Compute balanced class weights: weight inversely proportional to class frequency
total = len(y_train)
n_classes = 2
weight_0 = total / (n_classes * np.sum(y_train == 0))
weight_1 = total / (n_classes * np.sum(y_train == 1))
class_weights = {0: weight_0, 1: weight_1}

print("CLASS WEIGHTS (to address imbalance):")
print(f"  Weight for class 0 (no pepper): {weight_0:.4f}")
print(f"  Weight for class 1 (pepper):    {weight_1:.4f}")
print("-" * 70)

# -----------------------
# Train logistic regression with class weights
# -----------------------
print("Training logistic regression with balanced class weights...")
beta = logistic_regression_fit(X_train_i, y_train, class_weights=class_weights,
                                lr=0.5, max_iter=2000)

# Predictions
y_prob_test = predict_proba(X_test_i, beta)
y_pred_test = predict_class(X_test_i, beta, threshold=0.5)

# -----------------------
# Model quality metrics
# -----------------------
metrics = classification_metrics(y_test, y_pred_test, y_prob_test)

print("\nMODEL PERFORMANCE (Test Set):")
print("-" * 70)
print(f"AUC-ROC:     {metrics['auc_roc']:.4f}")
print(f"Accuracy:    {metrics['accuracy']:.4f}")
print(f"Precision:   {metrics['precision']:.4f}")
print(f"Recall:      {metrics['recall']:.4f}")
print(f"F1 Score:    {metrics['f1']:.4f}")
print(f"Specificity: {metrics['specificity']:.4f}")
print("-" * 70)

print("\nCONFUSION MATRIX:")
print(f"                 Predicted")
print(f"              No Pepper  Pepper")
print(f"Actual No Pepper   {metrics['tn']:5d}    {metrics['fp']:5d}")
print(f"Actual Pepper      {metrics['fn']:5d}    {metrics['tp']:5d}")
print("-" * 70)

# -----------------------
# Plot ROC Curve
# -----------------------
fpr, tpr, thresholds = compute_roc_curve(y_test, y_prob_test, n_thresholds=200)

fig, ax = plt.subplots(figsize=(8, 6))

# Plot ROC curve
ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {metrics["auc_roc"]:.4f})')

# Plot diagonal (random classifier)
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier (AUC = 0.5)')

# Formatting
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
ax.set_ylabel('True Positive Rate (Sensitivity/Recall)', fontsize=12)
ax.set_title('ROC Curve: Predicting Professor "Pepper" Status', fontsize=14)
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)

# Add annotation for the operating point at threshold=0.5
idx_05 = np.argmin(np.abs(thresholds - 0.5))
ax.scatter([fpr[idx_05]], [tpr[idx_05]], color='red', s=100, zorder=5, label=f'Threshold=0.5')
ax.annotate(f'Threshold=0.5\nTPR={tpr[idx_05]:.2f}, FPR={fpr[idx_05]:.2f}',
            xy=(fpr[idx_05], tpr[idx_05]), xytext=(fpr[idx_05]+0.15, tpr[idx_05]-0.15),
            fontsize=10, arrowprops=dict(arrowstyle='->', color='red'))

plt.tight_layout()
plt.savefig('plots/q10_roc_curve.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nROC curve saved to: plots/q10_roc_curve.png")
print("-" * 70)

# -----------------------
# Feature importance (standardized coefficients)
# -----------------------
coef_df = pd.DataFrame({
    "feature": feature_cols,
    "coef_abs": np.abs(beta[1:]),
    "coef": beta[1:]
}).sort_values("coef_abs", ascending=False)

print("\nTOP 10 MOST PREDICTIVE FEATURES (by |coefficient|):")
print(coef_df.head(10).to_string(index=False, formatters={"coef_abs": "{:.4f}".format, "coef": "{:.4f}".format}))
print("-" * 70)

# Also train without class weights for comparison
print("\n" + "=" * 70)
print("COMPARISON: Model WITHOUT class weights")
print("=" * 70)
beta_unweighted = logistic_regression_fit(X_train_i, y_train, class_weights=None,
                                           lr=0.5, max_iter=2000)
y_prob_uw = predict_proba(X_test_i, beta_unweighted)
y_pred_uw = predict_class(X_test_i, beta_unweighted, threshold=0.5)
metrics_uw = classification_metrics(y_test, y_pred_uw, y_prob_uw)

print(f"AUC-ROC:     {metrics_uw['auc_roc']:.4f} (vs {metrics['auc_roc']:.4f} weighted)")
print(f"Accuracy:    {metrics_uw['accuracy']:.4f} (vs {metrics['accuracy']:.4f} weighted)")
print(f"Recall:      {metrics_uw['recall']:.4f} (vs {metrics['recall']:.4f} weighted)")
print(f"Precision:   {metrics_uw['precision']:.4f} (vs {metrics['precision']:.4f} weighted)")
print("-" * 70)

# ==============================================================================
# Q10 ANALYSIS AND ANSWER
# ==============================================================================
#
# QUESTION: Build a classification model predicting whether a professor receives
# a "pepper" from all available factors (tags + numerical). Include AUC-ROC and
# address class imbalance concerns.
#
# ------------------------------------------------------------------------------
# MODEL: Logistic Regression
# ------------------------------------------------------------------------------
# Predictors: 6 numerical features + 20 tag features = 26 total predictors
# Target: pepper (binary: 0 = no pepper, 1 = pepper)
#
# ------------------------------------------------------------------------------
# MODEL RESULTS:
# ------------------------------------------------------------------------------
# - AUC-ROC:     0.7982  (good discriminative ability)
# - Accuracy:    0.7351  (73.5% correct predictions)
# - Precision:   0.7145  (71.5% of predicted peppers are correct)
# - Recall:      0.7883  (78.8% of actual peppers are detected)
# - F1 Score:    0.7496
# - Specificity: 0.6813  (68.1% of non-peppers correctly identified)
#
# Confusion Matrix:
#                      Predicted
#                   No Pepper  Pepper
# Actual No Pepper     404      189
# Actual Pepper        127      473
#
# The model has good overall performance with AUC-ROC ~0.80, indicating it can
# distinguish between professors who receive a pepper vs those who don't
# reasonably well.
#
# ------------------------------------------------------------------------------
# CLASS IMBALANCE ASSESSMENT:
# ------------------------------------------------------------------------------
# Class distribution:
#   - No pepper: 2965 (49.7%)
#   - Pepper:    2998 (50.3%)
#   - Imbalance ratio: 0.99:1
#
# FINDING: The classes are nearly PERFECTLY BALANCED (50/50 split).
# This is somewhat surprising but means class imbalance is NOT a concern.
#
# Techniques implemented to address potential imbalance (for robustness):
#   1. Stratified train/test split - maintains class proportions in both sets
#   2. Balanced class weights - weights inversely proportional to class frequency
#   3. Reporting multiple metrics (precision, recall, F1) not just accuracy
#
# Since classes are balanced, weighted and unweighted models perform nearly
# identically (as shown in the comparison output).
#
# ------------------------------------------------------------------------------
# MOST PREDICTIVE FEATURES (Top 10 by |coefficient|):
# ------------------------------------------------------------------------------
#   1. avg_rating:        +1.16  (higher ratings -> more likely pepper)
#   2. amazing_lectures:  +0.37  (amazing lectures -> more pepper)
#   3. inspirational:     +0.33  (inspirational -> more pepper)
#   4. avg_difficulty:    +0.24  (harder courses -> more pepper?*)
#   5. male_prof:         -0.18  (male profs less likely to get pepper)
#   6. caring:            +0.15  (caring -> more pepper)
#   7. would_take_again:  +0.14  (positive student sentiment -> pepper)
#   8. num_ratings:       -0.13  (more ratings -> less pepper?*)
#   9. group_projects:    -0.12  (group projects -> less pepper)
#  10. hilarious:         +0.12  (humor -> more pepper)
#
# *Note: Some effects may be confounded. The difficulty effect might reflect
# that charismatic professors can maintain high engagement even in harder courses.
#
# Key insight: avg_rating is by far the strongest predictor of receiving a
# pepper, suggesting that student perception of teaching quality and physical
# attractiveness ratings are correlated.
#
# ==============================================================================
