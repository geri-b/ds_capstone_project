"""
Q7: Regression model predicting avg_rating from all numerical predictors
in rmpCapstoneNum.csv, with R^2 and RMSE reported.

Collinearity note:
- The raw data includes both `male` and `female` indicators. After filtering to "clear gender"
  rows (exactly one of male/female == 1), these two columns are perfectly collinear
  (female = 1 - male). We therefore keep a single gender indicator `male_prof` and drop
  `male`/`female` from the regression predictors.
"""

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

SEED = 16784684


def train_test_split_indices(n: int, test_size: float = 0.2, seed: int = SEED):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(round(n * test_size))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return train_idx, test_idx


def standardize_train_test(X_train: np.ndarray, X_test: np.ndarray):
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0, ddof=0)
    # Guard against constant / near-constant columns (near-zero std can blow up z-scores).
    sigma = np.where(sigma < 1e-12, 1.0, sigma)
    return (X_train - mu) / sigma, (X_test - mu) / sigma, mu, sigma


def add_intercept(X: np.ndarray):
    return np.column_stack([np.ones(X.shape[0]), X])


def ols_fit(X: np.ndarray, y: np.ndarray):
    """
    Ordinary least squares using lstsq.
    X should already include intercept column if desired.
    Returns beta.
    """
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float = 1e-6):
    """
    Ridge regression closed-form solution:
      beta = (X'X + alpha*I)^(-1) X'y
    We do NOT penalize the intercept term (column 0) if present.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    p = X.shape[1]
    XtX = X.T @ X
    reg = np.eye(p)
    reg[0, 0] = 0.0  # do not penalize intercept
    A = XtX + alpha * reg
    b = X.T @ y
    return np.linalg.solve(A, b)


def predict(X: np.ndarray, beta: np.ndarray):
    return X @ beta


def r2_score(y_true: np.ndarray, y_pred: np.ndarray):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def rmse(y_true: np.ndarray, y_pred: np.ndarray):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def vif_table(X: np.ndarray, feature_names: list[str]):
    """
    Variance Inflation Factor (VIF) for each column in X (no intercept).
    VIF_j = 1 / (1 - R^2_j), where R^2_j comes from regressing X_j on other predictors.
    """
    X = np.asarray(X, dtype=float)
    vifs = []
    for j in range(X.shape[1]):
        yj = X[:, j]
        X_others = np.delete(X, j, axis=1)
        X_others_i = add_intercept(X_others)
        beta = ols_fit(X_others_i, yj)
        yhat = predict(X_others_i, beta)
        r2 = r2_score(yj, yhat)
        vif = 1.0 / (1.0 - r2) if (r2 is not None and r2 < 1.0) else np.inf
        vifs.append(vif)
    return pd.DataFrame({"feature": feature_names, "VIF": vifs}).sort_values("VIF", ascending=False)


# -----------------------
# Load + label columns (same convention as q1-3/q4)
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

# -----------------------
# Filter: clear gender + 10+ ratings (matches your earlier approach)
# -----------------------
df = num.loc[(num["male"] + num["female"] == 1)].copy()
df = df.loc[df["num_ratings"] >= 10].copy()
df["male_prof"] = df["male"]  # 1=male, 0=female

print("After filtering (clear gender + 10+ ratings):")
print("Total:", len(df))
print(df["male_prof"].value_counts().rename({1: "male", 0: "female"}))
print("-" * 70)

# -----------------------
# Build design matrix
# -----------------------
y = df["avg_rating"]

# Predictors: "all numerical predictors" except the outcome itself.
# Address collinearity by dropping the redundant gender dummy (`female`) and using `male_prof`.
feature_cols = [
    "avg_difficulty",
    "num_ratings",
    "pepper",
    "would_take_again",
    "num_online",
    "male_prof",
]
X = df[feature_cols]

# Drop rows with missing values in y or X
model_df = pd.concat([y, X], axis=1).dropna().copy()
y = model_df["avg_rating"].to_numpy(dtype=float)
X = model_df[feature_cols].to_numpy(dtype=float)

print(f"Rows used in regression after dropping missing values: {len(y)}")
print("-" * 70)

# -----------------------
# Train/test split + OLS fit
# -----------------------
train_idx, test_idx = train_test_split_indices(len(y), test_size=0.2, seed=SEED)
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

X_train_z, X_test_z, mu, sigma = standardize_train_test(X_train, X_test)

X_train_i = add_intercept(X_train_z)
X_test_i = add_intercept(X_test_z)

# Diagnostics: ill-conditioned design matrices can yield huge coefficients and numeric warnings.
cond = float(np.linalg.cond(X_train_i))
print(f"Design-matrix condition number (train): {cond:.3e}")

# Use a tiny ridge penalty to stabilize coefficients if needed (also helps with collinearity).
alpha = 1e-6
beta = ridge_fit(X_train_i, y_train, alpha=alpha)
yhat_test = predict(X_test_i, beta)

r2 = r2_score(y_test, yhat_test)
test_rmse = rmse(y_test, yhat_test)

print(f"Linear regression (ridge, alpha={alpha:g}) predicting avg_rating from numeric predictors")
print(f"Test R^2:  {r2:.4f}")
print(f"Test RMSE: {test_rmse:.4f} (rating points on the 0–5 scale)")
print("-" * 70)

# -----------------------
# Collinearity diagnostics (VIF on standardized training X)
# -----------------------
vif_df = vif_table(X_train_z, feature_cols)
print("VIF (Variance Inflation Factor) — higher means more collinearity:")
print(vif_df.to_string(index=False, formatters={"VIF": lambda v: f"{v:.2f}" if np.isfinite(v) else "inf"}))
print("-" * 70)

# -----------------------
# "Most strongly predictive" factor (standardized beta weights)
# Fit on TRAIN with standardized X and standardized y for comparable coefficients.
# -----------------------
y_train_z = (y_train - y_train.mean()) / (y_train.std(ddof=0) if y_train.std(ddof=0) != 0 else 1.0)
beta_std = ridge_fit(add_intercept(X_train_z), y_train_z, alpha=alpha)

coef_df = pd.DataFrame(
    {"feature": feature_cols, "std_beta_abs": np.abs(beta_std[1:]), "std_beta": beta_std[1:]}
).sort_values("std_beta_abs", ascending=False)

top = coef_df.iloc[0]
print("Standardized coefficients (trained on train split; larger |beta| => stronger predictor):")
print(coef_df.to_string(index=False, formatters={"std_beta_abs": "{:.4f}".format, "std_beta": "{:.4f}".format}))
print("-" * 70)
print(f"Most strongly predictive factor (by |standardized beta|): {top['feature']}")

# ==============================================================================
# Q7 ANALYSIS AND ANSWER
# ==============================================================================
#
# QUESTION: Build a regression model predicting average rating from all numerical
# predictors in rmpCapstoneNum.csv. Include R² and RMSE. Which factor is most
# strongly predictive? Address collinearity concerns.
#
# ------------------------------------------------------------------------------
# MODEL RESULTS:
# ------------------------------------------------------------------------------
# - Test R²:  0.8258  (model explains ~83% of variance in avg_rating)
# - Test RMSE: 0.3494 (average prediction error of ~0.35 points on 1-5 scale)
#
# ------------------------------------------------------------------------------
# MOST STRONGLY PREDICTIVE FACTOR:
# ------------------------------------------------------------------------------
# "would_take_again" is the most strongly predictive factor.
#
# Standardized coefficients (by magnitude):
#   1. would_take_again:  0.7090  (strongest positive predictor)
#   2. avg_difficulty:   -0.2033  (negative: harder classes → lower ratings)
#   3. pepper:            0.1222  (positive: "hot" professors rated higher)
#   4. male_prof:         0.0120  (negligible effect)
#   5. num_online:        0.0037  (negligible effect)
#   6. num_ratings:       0.0008  (negligible effect)
#
# Interpretation: Professors whose students say they "would take the class again"
# tend to have much higher ratings. Difficulty has a moderate negative effect.
#
# ------------------------------------------------------------------------------
# COLLINEARITY CONCERNS ADDRESSED:
# ------------------------------------------------------------------------------
# 1. The original data has both "male" and "female" columns. After filtering to
#    rows with clear gender (male + female == 1), these are perfectly collinear
#    (female = 1 - male). We keep only "male_prof" as a single gender indicator.
#
# 2. VIF (Variance Inflation Factor) was computed for all predictors:
#      - would_take_again: 1.66
#      - avg_difficulty:   1.42
#      - pepper:           1.27
#      - num_ratings:      1.02
#      - male_prof:        1.02
#      - num_online:       1.01
#
#    All VIF values are well below 5 (common threshold), indicating no
#    problematic multicollinearity remains after dropping the redundant
#    gender dummy variable.
#
# ==============================================================================


