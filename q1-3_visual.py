import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import levene

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

# Extract ratings
male_ratings = df_gender[df_gender["male_prof"] == 1]["avg_rating"]
female_ratings = df_gender[df_gender["male_prof"] == 0]["avg_rating"]

# Set style
sns.set(style="whitegrid", font_scale=1.2)

# ------
# 1. Overlaid KDE density plot
# ------
plt.figure(figsize=(10, 6))
sns.kdeplot(male_ratings, shade=True, label="Male", color="blue", linewidth=2)
sns.kdeplot(female_ratings, shade=True, label="Female", color="red", linewidth=2)

plt.title("Distribution of Average Ratings by Gender")
plt.xlabel("Average Rating")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.show()

# ------
# 2. Side-by-side histograms
# ------
plt.figure(figsize=(10, 6))
sns.histplot(male_ratings, bins=30, color="blue", label="Male", kde=False, stat="density", alpha=0.5)
sns.histplot(female_ratings, bins=30, color="red",  label="Female", kde=False, stat="density", alpha=0.5)

plt.title("Histogram of Professor Ratings by Gender")
plt.xlabel("Average Rating")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.show()

#Male-Female mean counts
print("Male-Female mean counts")
print(male_ratings.count())
print(female_ratings.count())
print("------------------------")

##Number of ratings visualization
max_ratings = 50  # cap for visualization

male_counts = df_gender[df_gender["male_prof"] == 1]["num_ratings"]
female_counts = df_gender[df_gender["male_prof"] == 0]["num_ratings"]

plt.figure(figsize=(12, 5))

# Female
plt.subplot(1, 2, 1)
plt.hist(female_counts[female_counts <= max_ratings],
         bins=50, edgecolor="black")
plt.title("Female Professors")
plt.xlabel("Number of ratings")
plt.ylabel("Number of professors")
plt.xlim(0, max_ratings)

# Male
plt.subplot(1, 2, 2)
plt.hist(male_counts[male_counts <= max_ratings],
         bins=50, edgecolor="black")
plt.title("Male Professors")
plt.xlabel("Number of ratings")
plt.ylabel("Number of professors")
plt.xlim(0, max_ratings)

plt.suptitle("Raw Distribution of Number of Ratings (≤ 50)", fontsize=14)
plt.tight_layout()
plt.show()
