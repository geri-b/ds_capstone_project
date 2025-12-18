##Confounders

import numpy as np
import statsmodels.formula.api as smf

import pandas as pd
from scipy.stats import ttest_ind, levene

# --- 1. Load CSVs ---
num = pd.read_csv("rmpCapstoneNum.csv", header=None)
qual = pd.read_csv("rmpCapstoneQual.csv", header=None)
tags = pd.read_csv("rmpCapstoneTags.csv", header=None)

# --- 2. Rename columns for clarity ---

num.columns = [
    "avg_rating",        # 1
    "avg_difficulty",    # 2
    "num_ratings",       # 3
    "pepper",            # 4
    "would_take_again",  # 5 (proportion 0-1)
    "num_online",        # 6
    "male",              # 7 (boolean 0/1)
    "female"             # 8 (boolean 0/1)
]

qual.columns = [
    "field",       # 1: Major/Field
    "university",  # 2
    "state"        # 3
]

tags.columns = [
    "tag_tough_grader",        # 1
    "tag_good_feedback",       # 2
    "tag_respected",           # 3
    "tag_lots_to_read",        # 4
    "tag_participation_matters", # 5
    "tag_dont_skip",           # 6
    "tag_lots_homework",       # 7
    "tag_inspirational",       # 8
    "tag_pop_quizzes",         # 9
    "tag_accessible",          # 10
    "tag_many_papers",         # 11
    "tag_clear_grading",       # 12
    "tag_hilarious",           # 13
    "tag_test_heavy",          # 14
    "tag_graded_few_things",   # 15
    "tag_amazing_lectures",    # 16
    "tag_caring",              # 17
    "tag_extra_credit",        # 18
    "tag_group_projects",      # 19
    "tag_lecture_heavy"        # 20
]

# --- 3. Merge into a single DataFrame (column-wise) ---

df = pd.concat([num, qual, tags], axis=1)

# Check gender combinations
print(df[["male", "female"]].value_counts(dropna=False))

# Keep only rows with a clear gender label:
# exactly one of male/female = 1
gender_mask = ((df["male"] == 1) & (df["female"] == 0)) | \
              ((df["male"] == 0) & (df["female"] == 1))

df_gender = df[gender_mask].copy()

# Define a single binary gender variable: 1 = male, 0 = female
df_gender["male_prof"] = df_gender["male"]
# Keep only needed columns
model_data = df_gender[
    [
        "avg_rating",
        "male_prof",
        "avg_difficulty",
        "num_ratings",
        "would_take_again",
        "num_online",
        "pepper",
        "field",
    ]
].copy()

# Log-transform number of ratings due to heavy skew
model_data["log_num_ratings"] = np.log1p(model_data["num_ratings"])

# Optional: total tags
tag_cols = [c for c in df_gender.columns if c.startswith("tag_")]
model_data["tag_total"] = df_gender[tag_cols].sum(axis=1)

# Remove rows with missing data
model_data = model_data.dropna()

# Regression formula
formula = """
avg_rating ~ male_prof
            + avg_difficulty
            + log_num_ratings
            + would_take_again
            + num_online
            + pepper
            + tag_total
            + C(field)
"""

model = smf.ols(formula=formula, data=model_data).fit()  # robust SE
print(model.summary())

#Is this statistically significant?
#Yes. p = 0.001 < 0.05.

#Is it practically meaningful?
#On a 5-point rating scale:
#The effect size is 0.0288 / 5 ≈ 0.6% of the scale.
#That is tiny in magnitude.
#It is over 10× smaller than effects from difficulty (–0.20) or pepper (+0.20).
#Thus:
#There is a statistically significant male advantage,
#But the effect size is extremely small and arguably not practically meaningful.


##Cohen's Q
male = df_gender[df_gender["male_prof"] == 1]["avg_rating"]
female = df_gender[df_gender["male_prof"] == 0]["avg_rating"]

n1, n2 = len(male), len(female)
s1, s2 = male.std(ddof=1), female.std(ddof=1)

s_pooled = np.sqrt(((n1 - 1)*s1**2 + (n2 - 1)*s2**2) / (n1 + n2 - 2))

# Raw Cohen's d
d_raw = (male.mean() - female.mean()) / s_pooled

# Regression-adjusted d
beta = model.params["male_prof"]
resid_sd = np.sqrt(model.mse_resid)
d_adjusted = beta / resid_sd

print(d_raw, d_adjusted)
