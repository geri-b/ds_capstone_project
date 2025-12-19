"""
Generate coefficient plots for Q7, Q8, Q9 to visualize predictor importance.
Uses seaborn for cleaner styling.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style (matching q1-3_visual.py)
sns.set(style="whitegrid", font_scale=1.2)

# Data from the model outputs (standardized coefficients)

# Q7: Numerical predictors -> Rating
q7_features = ['would_take_again', 'avg_difficulty', 'pepper', 'male_prof', 'num_online', 'num_ratings']
q7_coefs = [0.7090, -0.2033, 0.1222, 0.0120, 0.0037, 0.0008]

# Q8: Tags -> Rating
q8_features = ['tough_grader', 'amazing_lectures', 'good_feedback', 'caring', 'lecture_heavy',
               'group_projects', 'lots_homework', 'many_papers', 'graded_few_things', 'clear_grading',
               'pop_quizzes', 'accessible', 'hilarious', 'test_heavy', 'lots_to_read', 'respected',
               'participation_matters', 'extra_credit', 'inspirational', 'dont_skip']
q8_coefs = [-0.3990, 0.2138, 0.1652, 0.1616, -0.1497, -0.1065, -0.0966, -0.0881, -0.0785, 0.0673,
            0.0658, -0.0404, 0.0404, 0.0352, -0.0326, 0.0270, -0.0219, -0.0179, -0.0160, 0.0087]

# Q9: Tags -> Difficulty
q9_features = ['tough_grader', 'accessible', 'clear_grading', 'caring', 'hilarious',
               'dont_skip', 'pop_quizzes', 'graded_few_things', 'inspirational', 'extra_credit',
               'lots_to_read', 'many_papers', 'lecture_heavy', 'group_projects', 'test_heavy',
               'good_feedback', 'respected', 'amazing_lectures', 'lots_homework', 'participation_matters']
q9_coefs = [0.4688, 0.1999, -0.1681, -0.1502, -0.1223, 0.1107, -0.0836, -0.0602, -0.0600, -0.0584,
            0.0513, 0.0299, 0.0279, 0.0261, 0.0231, 0.0158, -0.0087, 0.0027, -0.0019, 0.0003]


def plot_coefficients(features, coefs, title, filename, figsize=(10, 6), palette="RdYlGn"):
    """Create a horizontal bar chart of coefficients using seaborn."""
    # Create dataframe for seaborn
    df = pd.DataFrame({
        'Feature': features,
        'Coefficient': coefs,
        'Abs_Coef': np.abs(coefs)
    })

    # Sort by absolute value
    df = df.sort_values('Abs_Coef', ascending=True)

    # Create color based on sign
    colors = ['#e74c3c' if c < 0 else '#27ae60' for c in df['Coefficient']]

    fig, ax = plt.subplots(figsize=figsize)

    # Create horizontal bar plot
    bars = ax.barh(df['Feature'], df['Coefficient'], color=colors, edgecolor='black', linewidth=0.5)

    # Add vertical line at 0
    ax.axvline(x=0, color='black', linewidth=1, linestyle='-')

    # Labels and title
    ax.set_xlabel('Standardized Coefficient', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add value annotations
    for bar, coef in zip(bars, df['Coefficient']):
        width = bar.get_width()
        label_x = width + 0.015 if width >= 0 else width - 0.015
        ha = 'left' if width >= 0 else 'right'
        ax.text(label_x, bar.get_y() + bar.get_height()/2, f'{coef:.3f}',
                va='center', ha=ha, fontsize=9, fontweight='bold')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#27ae60', edgecolor='black', label='Positive effect'),
                       Patch(facecolor='#e74c3c', edgecolor='black', label='Negative effect')]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'plots/{filename}', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: plots/{filename}")


# Generate individual plots
plot_coefficients(q7_features, q7_coefs,
                  'Q7: Predicting Rating from Numerical Predictors ($R^2$ = 0.826)',
                  'q7_coefficients.png', figsize=(10, 5))

plot_coefficients(q8_features, q8_coefs,
                  'Q8: Predicting Rating from Tags ($R^2$ = 0.495)',
                  'q8_coefficients.png', figsize=(10, 8))

plot_coefficients(q9_features, q9_coefs,
                  'Q9: Predicting Difficulty from Tags ($R^2$ = 0.426)',
                  'q9_coefficients.png', figsize=(10, 8))


# Create comparison plot for Q7 vs Q8
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Q7 subplot
df_q7 = pd.DataFrame({'Feature': q7_features, 'Coefficient': q7_coefs})
df_q7 = df_q7.sort_values('Coefficient', key=abs, ascending=True)
colors_q7 = ['#e74c3c' if c < 0 else '#27ae60' for c in df_q7['Coefficient']]

axes[0].barh(df_q7['Feature'], df_q7['Coefficient'], color=colors_q7, edgecolor='black', linewidth=0.5)
axes[0].axvline(x=0, color='black', linewidth=1)
axes[0].set_xlabel('Standardized Coefficient', fontsize=11)
axes[0].set_title('Q7: Numerical Predictors → Rating\n($R^2$ = 0.826)', fontsize=12, fontweight='bold')

for i, (feat, coef) in enumerate(zip(df_q7['Feature'], df_q7['Coefficient'])):
    label_x = coef + 0.02 if coef >= 0 else coef - 0.02
    ha = 'left' if coef >= 0 else 'right'
    axes[0].text(label_x, i, f'{coef:.3f}', va='center', ha=ha, fontsize=9, fontweight='bold')

# Q8 subplot (top 10)
df_q8 = pd.DataFrame({'Feature': q8_features, 'Coefficient': q8_coefs})
df_q8['Abs'] = np.abs(df_q8['Coefficient'])
df_q8 = df_q8.nlargest(10, 'Abs').sort_values('Abs', ascending=True)
colors_q8 = ['#e74c3c' if c < 0 else '#27ae60' for c in df_q8['Coefficient']]

axes[1].barh(df_q8['Feature'], df_q8['Coefficient'], color=colors_q8, edgecolor='black', linewidth=0.5)
axes[1].axvline(x=0, color='black', linewidth=1)
axes[1].set_xlabel('Standardized Coefficient', fontsize=11)
axes[1].set_title('Q8: Tags → Rating (Top 10)\n($R^2$ = 0.495)', fontsize=12, fontweight='bold')

for i, (feat, coef) in enumerate(zip(df_q8['Feature'], df_q8['Coefficient'])):
    label_x = coef + 0.015 if coef >= 0 else coef - 0.015
    ha = 'left' if coef >= 0 else 'right'
    axes[1].text(label_x, i, f'{coef:.3f}', va='center', ha=ha, fontsize=9, fontweight='bold')

# Add legend to figure
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#27ae60', edgecolor='black', label='Positive effect'),
                   Patch(facecolor='#e74c3c', edgecolor='black', label='Negative effect')]
fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=11, bbox_to_anchor=(0.5, -0.02))

plt.suptitle('Comparison: Numerical Predictors vs Tags for Predicting Rating', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('plots/q7_q8_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: plots/q7_q8_comparison.png")


# Create Q8 vs Q9 comparison (tough_grader effect)
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Q8 subplot (top 10)
df_q8 = pd.DataFrame({'Feature': q8_features, 'Coefficient': q8_coefs})
df_q8['Abs'] = np.abs(df_q8['Coefficient'])
df_q8 = df_q8.nlargest(10, 'Abs').sort_values('Abs', ascending=True)
colors_q8 = ['#e74c3c' if c < 0 else '#27ae60' for c in df_q8['Coefficient']]

axes[0].barh(df_q8['Feature'], df_q8['Coefficient'], color=colors_q8, edgecolor='black', linewidth=0.5)
axes[0].axvline(x=0, color='black', linewidth=1)
axes[0].set_xlabel('Standardized Coefficient', fontsize=11)
axes[0].set_title('Q8: Tags → Rating (Top 10)\n($R^2$ = 0.495)', fontsize=12, fontweight='bold')

for i, (feat, coef) in enumerate(zip(df_q8['Feature'], df_q8['Coefficient'])):
    label_x = coef + 0.015 if coef >= 0 else coef - 0.015
    ha = 'left' if coef >= 0 else 'right'
    axes[0].text(label_x, i, f'{coef:.3f}', va='center', ha=ha, fontsize=9, fontweight='bold')

# Q9 subplot (top 10)
df_q9 = pd.DataFrame({'Feature': q9_features, 'Coefficient': q9_coefs})
df_q9['Abs'] = np.abs(df_q9['Coefficient'])
df_q9 = df_q9.nlargest(10, 'Abs').sort_values('Abs', ascending=True)
colors_q9 = ['#e74c3c' if c < 0 else '#27ae60' for c in df_q9['Coefficient']]

axes[1].barh(df_q9['Feature'], df_q9['Coefficient'], color=colors_q9, edgecolor='black', linewidth=0.5)
axes[1].axvline(x=0, color='black', linewidth=1)
axes[1].set_xlabel('Standardized Coefficient', fontsize=11)
axes[1].set_title('Q9: Tags → Difficulty (Top 10)\n($R^2$ = 0.426)', fontsize=12, fontweight='bold')

for i, (feat, coef) in enumerate(zip(df_q9['Feature'], df_q9['Coefficient'])):
    label_x = coef + 0.015 if coef >= 0 else coef - 0.015
    ha = 'left' if coef >= 0 else 'right'
    axes[1].text(label_x, i, f'{coef:.3f}', va='center', ha=ha, fontsize=9, fontweight='bold')

# Add legend
fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=11, bbox_to_anchor=(0.5, -0.02))

plt.suptitle('Tags Predicting Rating vs Difficulty: Note "tough_grader" Sign Flip', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('plots/q8_q9_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: plots/q8_q9_comparison.png")

print("\nAll plots generated successfully!")
