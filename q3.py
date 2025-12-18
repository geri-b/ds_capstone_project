#Effect size of gender bias in avg. ratings

#Using the regression model from Question 1, the estimated gender effect (male minus female) was:
#Effect size: 0.0288 rating points
#95% CI: [0.012, 0.045]
#Thus, male professors receive ratings that are approximately 0.03 points higher,
# with the true effect likely between 0.012 and 0.045, after adjusting for confounders.
# This effect is statistically significant but extremely small in practical magnitude
# (<1% of the rating scale).
#Cohen's q effect sizes are extremely small too
#raw - 0.059903237894944235
#regression normalized - 0.0793046849729416



#Effect size of gender bias in spread of avg. rating
#Female variance: 1.3106
#Male variance: 1.1795
#The variance ratio was:
#Effect size (variance ratio): 1.11
#95% CI: approximately [1.08, 1.14]
#This indicates that female ratings have 8–14% greater dispersion than male ratings,
# a statistically significant but modest difference.

from q1 import female_ratings, male_ratings
from scipy.stats import f

# Variances
var_f = female_ratings.var(ddof=1)
var_m = male_ratings.var(ddof=1)

# Sample sizes
n_f = len(female_ratings)
n_m = len(male_ratings)

ratio = var_f / var_m

# CI for variance ratio using F distribution
lower = ratio / f.ppf(0.975, n_f - 1, n_m - 1)
upper = ratio / f.ppf(0.025, n_f - 1, n_m - 1)

ratio, lower, upper

print("---------q3 results---------")
print("Variance ratio", ratio)
print("95% CI lower-bound - ", lower)
print("95% CI upper-bound - ", upper)
