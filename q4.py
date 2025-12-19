import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
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

# -----------------------
# Filter: clear gender + 10+ ratings only to avoid biased tags
# -----------------------
df = df.loc[(df["male"] + df["female"] == 1)].copy()
df["male_prof"] = df["male"]  # 1=male, 0=female
df = df.loc[df["num_ratings"] >= 10].copy()

print("After filtering (clear gender + 10+ ratings):")
print("Total:", len(df))
print(df["male_prof"].value_counts().rename({1: "male", 0: "female"}))
print("-" * 60)

# -----------------------
# Helper: avoid scientific notation for p-values
# -----------------------
def fmt_p(p):
    # prints decimals (no scientific notation) while staying readable
    if p < 1e-10:
        return f"{p:.12f}"
    return f"{p:.10f}"

# -----------------------
# Chi-square per tag (binary: received tag at least once)
# -----------------------
results = []

for tag in tags.columns:
    tag_yes = (df[tag].fillna(0) > 0).astype(int)

    # 2x2: rows = gender (0 female, 1 male), cols = tag (0 no, 1 yes)
    table = pd.crosstab(df["male_prof"], tag_yes).reindex(index=[0, 1], columns=[0, 1], fill_value=0)

    chi2, p, dof, expected = chi2_contingency(table.values, correction=False)

    f_yes = table.loc[0, 1]
    f_n   = table.loc[0].sum()
    m_yes = table.loc[1, 1]
    m_n   = table.loc[1].sum()

    p_f = f_yes / f_n if f_n else np.nan
    p_m = m_yes / m_n if m_n else np.nan
    diff = p_m - p_f  # male - female probability of receiving tag

    results.append({
        "tag": tag.replace("tag_", "").replace("_", " "),
        "chi2": chi2,
        "p_value": p,
        "female_yes": int(f_yes),
        "female_total": int(f_n),
        "female_rate": p_f,
        "male_yes": int(m_yes),
        "male_total": int(m_n),
        "male_rate": p_m,
        "rate_diff_M_minus_F": diff
    })

res = pd.DataFrame(results).sort_values("p_value").reset_index(drop=True)

# -----------------------
# Print results table to compare hand by hand
# -----------------------
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)

out = res.copy()
out["chi2"] = out["chi2"].map(lambda x: float(f"{x:.4f}"))
out["female_rate"] = out["female_rate"].map(lambda x: float(f"{x:.6f}"))
out["male_rate"] = out["male_rate"].map(lambda x: float(f"{x:.6f}"))
out["rate_diff_M_minus_F"] = out["rate_diff_M_minus_F"].map(lambda x: float(f"{x:.6f}"))
out["p_value"] = out["p_value"].map(lambda x: float(f"{x:.12f}"))  # force decimal form

print("Chi-square tests per tag (binary: received ≥1 vs 0):")
print("(Rates are per-professor probabilities; Diff = Male rate - Female rate)")
print("-" * 60)

print(out[[
    "tag", "chi2", "p_value",
    "female_yes", "female_total", "female_rate",
    "male_yes", "male_total", "male_rate",
    "rate_diff_M_minus_F"
]].to_string(index=False))

print("-" * 60)

# -----------------------
# Significant tags + most/least gendered
# -----------------------
alpha = 0.005
sig = res[res["p_value"] < alpha]

print(f"Significant tags (p < {alpha}): {len(sig)} / 20")

print("\nTop 3 most gendered tags (lowest p-value):")
for _, r in res.head(3).iterrows():
    print(f"- {r['tag']}: p={fmt_p(r['p_value'])}, "
          f"rates M={r['male_rate']:.4f} vs F={r['female_rate']:.4f} (diff={r['rate_diff_M_minus_F']:.4f})")

print("\nTop 3 least gendered tags (highest p-value):")
for _, r in res.tail(3).sort_values("p_value", ascending=False).iterrows():
    print(f"- {r['tag']}: p={fmt_p(r['p_value'])}, "
          f"rates M={r['male_rate']:.4f} vs F={r['female_rate']:.4f} (diff={r['rate_diff_M_minus_F']:.4f})")

# -----------------------
# Visual: Male vs Female tag rates with diff (Male-Female)
# -----------------------
plot_df = res.copy().sort_values("rate_diff_M_minus_F")  # sort by effect direction/size
x = np.arange(len(plot_df))

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x, plot_df["rate_diff_M_minus_F"])
ax.axhline(0, linewidth=1)

ax.set_xticks(x)
ax.set_xticklabels(plot_df["tag"], rotation=70, ha="right")
ax.set_ylabel("Rate difference (Male - Female)")
ax.set_title("Gender differences in tag rates (received ≥1 tag)")
plt.tight_layout()
plt.show()

# Top 3 most gendered tags (lowest p-value):
# - hilarious: p=0.000000000000, rates M=0.6850 vs F=0.4923 (diff=0.1927)
# - participation matters: p=0.000000000000, rates M=0.7213 vs F=0.7970 (diff=-0.0756)
# - group projects: p=0.000000000008, rates M=0.2548 vs F=0.3287 (diff=-0.0739)
#
# Top 3 least gendered tags (highest p-value):
# - pop quizzes: p=0.7517816077, rates M=0.2242 vs F=0.2274 (diff=-0.0032)
# - clear grading: p=0.3592825232, rates M=0.7617 vs F=0.7710 (diff=-0.0093)
# - accessible: p=0.1560741626, rates M=0.5867 vs F=0.5699 (diff=0.0167)