import pandas as pd
from scipy.stats import ttest_ind, levene, mannwhitneyu

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

##Levene's test to check if variance is high
male_ratings = df_gender[df_gender["male_prof"] == 1]["avg_rating"]
female_ratings = df_gender[df_gender["male_prof"] == 0]["avg_rating"]

stat, p = levene(male_ratings, female_ratings)

print("Levene's test statistic:", stat)
print("p-value:", p)
##After running test, p value extremely small, almost zero, thus high variance
##This also answers question 2


## Run Welch's test, high variance between sets
t_stat, p_value = ttest_ind(
    male_ratings,
    female_ratings,
    equal_var=False  # Welch's t-test
)

print("------------------------")
print("Welch t-test results:")
print("t-statistic:", t_stat)
print("p-value:", p_value)
print("Difference in means:", male_ratings.mean() - female_ratings.mean())

##After running code
#p-value: 9.588557475706342e-12
#At first sight, given avg. rating of males is higher
#and p value almost zero, seems plausible to have some bias
#Difference in means though is very  small (3.811, 3.878), still not convincing

##Run Mann Whitney test
u_stat, p_value = mannwhitneyu(
    male_ratings,
    female_ratings,
    alternative='two-sided'
)

print("------------------------")
print("Mann Whitney u-test results:")
print("u-statistic:", u_stat)
print("p-value:", p_value)
print("Difference in means:", male_ratings.mean() - female_ratings.mean())
#Mann Whitney u-test results:
#u-statistic: 346129258.5
#p-value: 8.547294844039601e-06
#Difference in means: 0.06676637610792913

