# Gender bias: central tendency + spread with BOOTSTRAP CIs (no Welch/F-based CIs)
# Keeps only clear gender labels and professors with 10+ ratings.

import numpy as np
import pandas as pd
from scipy.stats import levene, ttest_ind, mannwhitneyu
SEED = 16784684

# -----------------------
# Load + label columns
# -----------------------
num  = pd.read_csv("rmpCapstoneNum.csv",  header=None)
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

# Show gender labeling patterns
print(df[["male", "female"]].value_counts(dropna=False))

# -----------------------
# Keep only clear gender labels (exactly one of male/female == 1)
# and apply 10+ ratings threshold BEFORE everything else
# -----------------------
df_gender = df.loc[(df["male"] + df["female"] == 1)].copy()
df_gender["male_prof"] = df_gender["male"]  # 1 = male, 0 = female

df10 = df_gender.loc[df_gender["num_ratings"] >= 10].copy()

print("\nAfter filtering for professors with 10+:")
print("Total:", len(df10))
print(df10["male_prof"].value_counts())
print("------------------------")

# -----------------------
# Bootstrap CI helper (percentile bootstrap)
# -----------------------
def bootstrap_ci(stat_fn, m, f, B=5000, alpha=0.05, seed=SEED):
    rng = np.random.default_rng(seed)
    m = np.asarray(m); f = np.asarray(f)
    n1, n2 = m.size, f.size

    boots = np.empty(B)
    for b in range(B):
        mb = m[rng.integers(0, n1, n1)]
        fb = f[rng.integers(0, n2, n2)]
        boots[b] = stat_fn(mb, fb)

    lo = np.quantile(boots, alpha/2)
    hi = np.quantile(boots, 1 - alpha/2)
    return (lo, hi)

# -----------------------
# Tests + bootstrap CIs
# -----------------------
def run_tests_with_bootstrap(data, outcome="avg_rating", B=5000, seed=SEED):
    m = data.loc[data["male_prof"] == 1, outcome].dropna().to_numpy()
    f = data.loc[data["male_prof"] == 0, outcome].dropna().to_numpy()

    diff_mean = m.mean() - f.mean()
    diff_median = np.median(m) - np.median(f)

    var_m = m.var(ddof=1)
    var_f = f.var(ddof=1)
    var_ratio = var_f / var_m  # (female / male)

    # Tests (optional, but keeps your current outputs)
    lev_stat, lev_p = levene(m, f)
    t_stat, t_p = ttest_ind(m, f, equal_var=False)  # Welch (optional)
    u_stat, u_p = mannwhitneyu(m, f, alternative="two-sided")

    # Bootstrap 95% CIs (percentile)
    ci_mean = bootstrap_ci(lambda a,b: a.mean() - b.mean(), m, f, B=B, seed=seed)
    ci_median = bootstrap_ci(lambda a,b: np.median(a) - np.median(b), m, f, B=B, seed=seed)
    ci_vr = bootstrap_ci(lambda a,b: b.var(ddof=1) / a.var(ddof=1), m, f, B=B, seed=seed)

    print(f"Counts (M/F): {m.size} / {f.size}")
    print(f"Means  (M/F): {m.mean():.4f} / {f.mean():.4f}  | Diff (M-F): {diff_mean:.6f}")
    print(f"Medians(M/F): {np.median(m):.4f} / {np.median(f):.4f}  | Diff (M-F): {diff_median:.6f}")
    print(f"Variances (M/F): {var_m:.4f} / {var_f:.4f} | Var ratio (F/M): {var_ratio:.6f}")

    print(f"Levene: stat={lev_stat:.4f}, p={lev_p:.10f}")
    print(f"Welch t-test (optional): t={t_stat:.4f}, p={t_p:.10f}")
    print(f"Mann–Whitney: U={u_stat:.1f}, p={u_p:.10f}")

    print("\n--- 95% Bootstrap CIs (percentile) ---")
    print("Mean diff (M-F):   ({:.6f}, {:.6f})".format(ci_mean[0], ci_mean[1]))
    print("Median diff (M-F): ({:.6f}, {:.6f})".format(ci_median[0], ci_median[1]))
    print("Var ratio (F/M):   ({:.6f}, {:.6f})".format(ci_vr[0], ci_vr[1]))

    return m, f  # return arrays for Cohen's d

# -----------------------
# Run (filtered dataset)
# -----------------------
male_arr, female_arr = run_tests_with_bootstrap(df10, outcome="avg_rating", B=5000, seed=SEED)

# -----------------------
# Cohen's d (raw) using df10 (10+ ratings)
# -----------------------
n1, n2 = male_arr.size, female_arr.size
s1, s2 = male_arr.std(ddof=1), female_arr.std(ddof=1)
s_pooled = np.sqrt(((n1 - 1)*s1**2 + (n2 - 1)*s2**2) / (n1 + n2 - 2))
d_raw = (male_arr.mean() - female_arr.mean()) / s_pooled

print("\nCohen's d (raw, df10): {:.6f}".format(d_raw))

#Question 1
# #p-val is below <0.005, but we cannot conclude there is bias as we've not considered the confounds
#Null hypothesis is dropped, at first sight there are grounds to doubt that male professors received
#higher ratings

#Question 2
#Yes, there is some degree of difference in the variance with female professors having slightly
#more dispersed ratings 
#According to Levene's (suggested by ChatGPT and researched from us) test p-val < 0.005
#But again, a ~10% increase in variance is modest and furthermore distributions from visualization
#part look similar

#Question 3
#Central Tendency
#95% Bootstrap CIs  - Mean difference CI: (0.035, 0.117) | Median difference CI: (0.000, 0.200)
#The effect is positive, small, and precisely estimated
#On a 0–5 scale, this is roughly 1.5–2.5% of the scale
#The likely size of the pro-male rating difference is approximately 0.04–0.12 points on a 5-point scale.

#Variance
#Variance ratio (F/M): 1.098
#95% Bootstrap CI - (1.022, 1.181)
#Female professors’ ratings are approximately 2–18% more variable than those of male professors.
# This is statistically significant, but practically a modest effect size



