import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

num = pd.read_csv("rmpCapstoneNum.csv", header=None)
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
df = df.loc[df["num_ratings"] >= 10].copy()

# Keep only English vs Computer Science
fields_keep = ["English", "Computer Science"]
df_sub = df.loc[df["field"].isin(fields_keep), ["field", "tag_tough_grader"]].dropna().copy()

# Binary: received at least one "tough grader" tag
df_sub["tough_binary"] = (df_sub["tag_tough_grader"] > 0).astype(int)

# 2x2 contingency table: rows=field, cols=tough(0/1)
tab = pd.crosstab(df_sub["field"], df_sub["tough_binary"])
tab = tab.reindex(index=fields_keep, columns=[0, 1], fill_value=0)  # enforce order

chi2, p, dof, expected = chi2_contingency(tab)

# Proportions
counts = tab.sum(axis=1)
props = tab[1] / counts
diff = props.loc["English"] - props.loc["Computer Science"]

# Print results
pd.set_option("display.float_format", lambda x: f"{x:.6f}")

print("\nContingency table (0=No tough tag, 1=Has tough tag):")
print(tab)

print("\nProportion with 'Tough grader' tag:")
print(props.rename("prop_tough"))

print(f"\nDifference in proportions (English - CS): {diff:.6f}")
print(f"Chi-square test: chi2={chi2:.4f}, dof={dof}, p={p:.10f}")

# Visual: bar chart of proportions
plt.figure(figsize=(6, 3))
plt.bar(props.index, props.values)
plt.ylim(0, max(props.values) * 1.2 if max(props.values) > 0 else 1)
plt.ylabel("Proportion with 'Tough grader' tag")
plt.title("'Tough grader' prevalence: English vs Computer Science")
plt.tight_layout()
plt.show()
