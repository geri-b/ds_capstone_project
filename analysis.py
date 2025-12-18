import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

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
df_gender["male_prof"] = df_gender["male"]  # since they are mutually exclusive now

# Quick sanity check: counts by gender
print(df_gender["male_prof"].value_counts())

# Helper summarizer by gender
def summarize_by_gender(var):
    return df_gender.groupby("male_prof")[var].agg(["mean", "std", "var", "count"])

# 1 = male, 0 = female
print("Average rating by gender:")
print(summarize_by_gender("avg_rating"), "\n")

print("Average difficulty by gender:")
print(summarize_by_gender("avg_difficulty"), "\n")

print("Number of ratings by gender:")
print(summarize_by_gender("num_ratings"), "\n")

print("% Pepper by gender:")
print(
    df_gender.groupby("male_prof")["pepper"]
    .mean()
    .rename("pepper_rate")
)

print("\n'Would take again' proportion by gender:")
print(summarize_by_gender("would_take_again"), "\n")

