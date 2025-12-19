import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
SEED = 16784684


num  = pd.read_csv("rmpCapstoneNum.csv", header=None)
qual = pd.read_csv("rmpCapstoneQual.csv", header=None)
tags = pd.read_csv("rmpCapstoneTags.csv", header=None)

num.columns  = ["avg_rating","avg_difficulty","num_ratings","pepper","would_take_again","num_online","male","female"]
qual.columns = ["field","university","state"]
tags.columns = [
    "tag_tough_grader","tag_good_feedback","tag_respected","tag_lots_to_read","tag_participation_matters",
    "tag_dont_skip","tag_lots_homework","tag_inspirational","tag_pop_quizzes","tag_accessible",
    "tag_many_papers","tag_clear_grading","tag_hilarious","tag_test_heavy","tag_graded_few_things",
    "tag_amazing_lectures","tag_caring","tag_extra_credit","tag_group_projects","tag_lecture_heavy"
]

df = pd.concat([num, qual, tags], axis=1)

df = df.loc[(df["male"] + df["female"] == 1)].copy()
df["male_prof"] = df["male"]  # 1 = male, 0 = female
df = df.loc[df["num_ratings"] >= 10].copy()

df = df.dropna(subset=["avg_difficulty"])

male_diff   = df.loc[df["male_prof"] == 1, "avg_difficulty"]
female_diff = df.loc[df["male_prof"] == 0, "avg_difficulty"]

u_stat, p_value = mannwhitneyu(
    male_diff,
    female_diff,
    alternative="two-sided"
)

print("Counts (M/F):", male_diff.size, "/", female_diff.size)
print("Means  (M/F):", round(male_diff.mean(), 4), "/", round(female_diff.mean(), 4))
print("Medians(M/F):", round(male_diff.median(), 4), "/", round(female_diff.median(), 4))
print("Mann–Whitney U statistic:", u_stat)
print("p-value:", f"{p_value:.10f}")

#No gender difference in regard to average difficulty
#p-value: 0.786; null hypothesis stands

#Question 6 | Quantifying effect size
# Bootstrapping since using Mann Whitney U
def bootstrap_ci(stat_fn, m, f, B=5000, alpha=0.05, seed=SEED):
    rng = np.random.default_rng(seed)
    m = np.asarray(m)
    f = np.asarray(f)

    boots = np.empty(B)
    for b in range(B):
        mb = m[rng.integers(0, len(m), len(m))]
        fb = f[rng.integers(0, len(f), len(f))]
        boots[b] = stat_fn(mb, fb)

    return (
        np.quantile(boots, alpha / 2),
        np.quantile(boots, 1 - alpha / 2)
    )

# Effect sizes
mean_diff   = male_diff.mean() - female_diff.mean()
median_diff = male_diff.median() - female_diff.median()

# 95% bootstrap CIs
ci_mean   = bootstrap_ci(lambda a, b: a.mean() - b.mean(), male_diff, female_diff)
ci_median = bootstrap_ci(lambda a, b: np.median(a) - np.median(b), male_diff, female_diff)

print("\nDifficulty difference (Male - Female)")
print(f"Mean difference:   {mean_diff:.4f}")
print(f"95% CI (mean):     ({ci_mean[0]:.4f}, {ci_mean[1]:.4f})")

print(f"Median difference: {median_diff:.4f}")
print(f"95% CI (median):   ({ci_median[0]:.4f}, {ci_median[1]:.4f})")

#VISUALIZATION
# Values from your results
mean_diff = -0.0070
ci_low, ci_high = -0.0432, 0.0284

plt.figure(figsize=(6, 2))
plt.errorbar(
    x=mean_diff,
    y=0,
    xerr=[[mean_diff - ci_low], [ci_high - mean_diff]],
    fmt='o',
    capsize=5
)

plt.axvline(0, linestyle='--', linewidth=1)
plt.yticks([])
plt.xlabel("Mean difference in average difficulty (Male − Female)")
plt.title("95% CI for Gender Difference in Average Difficulty")

plt.tight_layout()
plt.show()
##The estimated difference is almost zero.
# The confidence interval includes 0 making even the extremes of the CI very small (≈ ±0.04 on a 1–5 scale).
#Even the medians are pretty much the same
#NHTS stands - there is no meaningful gender difference in average difficulty ratings.