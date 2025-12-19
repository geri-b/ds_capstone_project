"""
Q9: Regression model predicting avg_difficulty from all tags in rmpCapstoneTags.csv,
with R^2 and RMSE reported.

Collinearity note:
- With 20 tag predictors, we check VIF to identify any problematic multicollinearity.
"""

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


def train_test_split_indices(n: int, test_size: float = 0.2, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(round(n * test_size))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return train_idx, test_idx


def standardize_train_test(X_train: np.ndarray, X_test: np.ndarray):
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0, ddof=0)
    sigma = np.where(sigma < 1e-12, 1.0, sigma)
    return (X_train - mu) / sigma, (X_test - mu) / sigma, mu, sigma


def add_intercept(X: np.ndarray):
    return np.column_stack([np.ones(X.shape[0]), X])


def ols_fit(X: np.ndarray, y: np.ndarray):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float = 1e-6):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    p = X.shape[1]
    XtX = X.T @ X
    reg = np.eye(p)
    reg[0, 0] = 0.0
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
# Filter: clear gender + 10+ ratings (consistent with Q7/Q8)
# -----------------------
df = df.loc[(df["male"] + df["female"] == 1)].copy()
df = df.loc[df["num_ratings"] >= 10].copy()

print("After filtering (clear gender + 10+ ratings):")
print("Total:", len(df))
print("-" * 70)

# -----------------------
# Build design matrix - predicting avg_difficulty
# -----------------------
y = df["avg_difficulty"]

# Predictors: all 20 tag columns
feature_cols = list(tags.columns)
X = df[feature_cols]

# Drop rows with missing values in y or X
model_df = pd.concat([y, X], axis=1).dropna().copy()
y = model_df["avg_difficulty"].to_numpy(dtype=float)
X = model_df[feature_cols].to_numpy(dtype=float)

print(f"Rows used in regression after dropping missing values: {len(y)}")
print("-" * 70)

# -----------------------
# Train/test split + Ridge fit
# -----------------------
train_idx, test_idx = train_test_split_indices(len(y), test_size=0.2, seed=0)
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

X_train_z, X_test_z, mu, sigma = standardize_train_test(X_train, X_test)

X_train_i = add_intercept(X_train_z)
X_test_i = add_intercept(X_test_z)

cond = float(np.linalg.cond(X_train_i))
print(f"Design-matrix condition number (train): {cond:.3e}")

alpha = 1e-6
beta = ridge_fit(X_train_i, y_train, alpha=alpha)
yhat_test = predict(X_test_i, beta)

r2 = r2_score(y_test, yhat_test)
test_rmse = rmse(y_test, yhat_test)

print(f"Linear regression (ridge, alpha={alpha:g}) predicting avg_difficulty from tags")
print(f"Test R^2:  {r2:.4f}")
print(f"Test RMSE: {test_rmse:.4f} (difficulty points on the 1-5 scale)")
print("-" * 70)

# -----------------------
# Collinearity diagnostics (VIF on standardized training X)
# -----------------------
vif_df = vif_table(X_train_z, feature_cols)
print("VIF (Variance Inflation Factor) - higher means more collinearity:")
print(vif_df.to_string(index=False, formatters={"VIF": lambda v: f"{v:.2f}" if np.isfinite(v) else "inf"}))
print("-" * 70)

high_vif = vif_df[vif_df["VIF"] > 5]
if len(high_vif) > 0:
    print(f"WARNING: {len(high_vif)} features have VIF > 5 (potential collinearity):")
    print(high_vif.to_string(index=False))
else:
    print("All VIF values are below 5 - no problematic multicollinearity detected.")
print("-" * 70)

# -----------------------
# "Most strongly predictive" factor (standardized beta weights)
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
print(f"Most strongly predictive tag (by |standardized beta|): {top['feature']}")

# ==============================================================================
# Q9 ANALYSIS AND ANSWER
# ==============================================================================
#
# QUESTION: Build a regression model predicting average difficulty from all tags
# in rmpCapstoneTags.csv. Include R^2 and RMSE. Which tag is most strongly
# predictive? Address collinearity concerns.
#
# ------------------------------------------------------------------------------
# MODEL RESULTS:
# ------------------------------------------------------------------------------
# - Test R^2:  0.3035  (model explains ~30% of variance in avg_difficulty)
# - Test RMSE: 0.6534  (average prediction error of ~0.65 points on 1-5 scale)
#
# ------------------------------------------------------------------------------
# MOST STRONGLY PREDICTIVE TAG:
# ------------------------------------------------------------------------------
# "tough_grader" is the most strongly predictive tag (std beta = +0.4524).
#
# Top 5 most predictive tags (by |standardized beta|):
#   1. tough_grader:       0.4524  (positive: tough graders = higher difficulty)
#   2. accessible:         0.1865  (positive: accessible profs rated as harder?*)
#   3. clear_grading:     -0.1576  (negative: clear grading = lower difficulty)
#   4. caring:            -0.1410  (negative: caring = lower perceived difficulty)
#   5. hilarious:         -0.0960  (negative: humor = lower perceived difficulty)
#
# *Note: The positive coefficient for "accessible" is surprising and may reflect
# that accessible professors have more student interaction, making students more
# aware of course demands, or it could be a spurious correlation.
#
# Interpretation: The "tough grader" tag is by far the strongest predictor of
# perceived difficulty. Tags indicating approachable teaching styles (clear
# grading, caring, hilarious) predict lower difficulty ratings.
#
# ------------------------------------------------------------------------------
# COLLINEARITY CONCERNS ADDRESSED:
# ------------------------------------------------------------------------------
# VIF computed for all 20 tag predictors:
#   - Highest VIF: "respected" (3.66)
#   - All VIF values are below 5 (common threshold)
#   - No problematic multicollinearity detected
#
# Note: VIF values are identical to Q8 since the same predictors (X) are used;
# only the target variable (y) changed from rating to difficulty.
#
# ------------------------------------------------------------------------------
# COMPARISON WITH Q8 (Tags predicting Rating):
# ------------------------------------------------------------------------------
#                        Q8 (Rating)     Q9 (Difficulty)
#   R^2:                 0.4264          0.3035
#   RMSE:                0.6793          0.6534
#
# Tags are somewhat better at predicting ratings (43%) than difficulty (30%).
# Interestingly, "tough_grader" is the top predictor for BOTH outcomes:
#   - Q8: tough_grader -> lower ratings  (std beta = -0.37)
#   - Q9: tough_grader -> higher difficulty (std beta = +0.45)
#
# This makes intuitive sense: tough grading leads to both higher perceived
# difficulty and lower satisfaction ratings.
#
# ==============================================================================
