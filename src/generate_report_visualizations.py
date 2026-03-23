import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set headless backend
plt.switch_backend('Agg')

# Ensure output directory exists
os.makedirs('output', exist_ok=True)

# 1. The MOOC Attrition "L-Curve" (Figure 1)
stages = ['Registration', 'Week 0 (Logged In)', 'Week 4', 'Week 8', 'Certification']
percentages = [100.0, 55.0, 25.0, 12.0, 5.5]

plt.figure(figsize=(10, 6))
plt.plot(stages, percentages, marker='o', color='#e63946', linewidth=3, markersize=8)
plt.title('The MOOC Attrition "L-Curve"', fontsize=14, fontweight='bold')
plt.ylabel('Percentage of Registered Users (%)')
plt.xlabel('Course Milestone')
plt.ylim(0, 105)
plt.grid(True, linestyle='--', alpha=0.6)

plt.annotate('No-Show Gap (45%)', xy=(1, 55), xytext=(2.5, 80), # Adjusted xy indexing for string categorical vs numeric
             arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10)
# Wait, xy=('Week 0 (Logged In)', 55) works in matplotlib 3+, but just in case, I'll use the user's string
plt.clf() # reset

plt.figure(figsize=(10, 6))
plt.plot(stages, percentages, marker='o', color='#e63946', linewidth=3, markersize=8)
plt.title('The MOOC Attrition "L-Curve"', fontsize=14, fontweight='bold')
plt.ylabel('Percentage of Registered Users (%)')
plt.xlabel('Course Milestone')
plt.ylim(0, 105)
plt.grid(True, linestyle='--', alpha=0.6)

plt.annotate('No-Show Gap (45%)', xy=('Week 0 (Logged In)', 55), xytext=(0.5, 80),
             arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10)

plt.tight_layout()
plt.savefig('output/figure1_l_curve.png', dpi=300)
plt.close()

# 2. Course Difficulty Distribution
difficulties = ['Beginner', 'Mixed', 'Intermediate', 'Advanced']
counts = [450, 100, 300, 150] 

plt.figure(figsize=(8, 5))
sns.barplot(x=difficulties, y=counts, palette='mako')
plt.title('Distribution of Course Difficulty Levels', fontsize=12, fontweight='bold')
plt.ylabel('Number of Courses')
plt.xlabel('Difficulty Tag')
plt.tight_layout()
plt.savefig('output/figure2_difficulty_dist.png', dpi=300)
plt.close()

# 3. Synthetic Rating Distribution Profile (Gaussian)
np.random.seed(42)

ratings = np.random.normal(3.0, 0.8, 1000)
ratings = np.clip(ratings, 1, 5) 

plt.figure(figsize=(8, 5))
plt.hist(ratings, bins=15, color='#457b9d', edgecolor='white', alpha=0.8)
plt.title('Synthetic Interaction Rating Profile', fontsize=12, fontweight='bold')
plt.xlabel('Star Rating (1-5)')
plt.ylabel('Frequency')
plt.axvline(np.mean(ratings), color='red', linestyle='dashed', linewidth=1, label='Mean Rating')
plt.legend()
plt.tight_layout()
plt.savefig('output/figure3_rating_profile.png', dpi=300)
plt.close()

# 4. The RMSE Paradox: Sacrificing Accuracy for Pedagogical Relevance
models = ['SVD (Baseline)', 'EduRec (Hybrid)']
rmse_scores = [0.85, 1.20]   
ndcg_scores = [0.81, 0.845]  

x = np.arange(len(models))
width = 0.35

fig, ax1 = plt.subplots(figsize=(10, 6))

rects1 = ax1.bar(x - width/2, rmse_scores, width, label='RMSE (Error Rate)', color='#adb5bd')
ax1.set_ylabel('RMSE (Lower is more accurate)')
ax1.set_ylim(0, 1.5)

ax2 = ax1.twinx()
rects2 = ax2.bar(x + width/2, ndcg_scores, width, label='NDCG@10 (Ranking Quality)', color='#1d3557')
ax2.set_ylabel('NDCG@10 (Higher is more relevant)')
ax2.set_ylim(0.75, 0.90)

plt.title('The RMSE Paradox: Sacrificing Accuracy for Pedagogical Relevance', fontsize=12, fontweight='bold')
plt.xticks(x, models)

ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig('output/figure4_evaluation_metrics.png', dpi=300)
plt.close()

# 5. Persona Performance Breakdown
personas = ['Beginner Web Dev', 'Advanced AI Researcher']
svd_performance = [0.78, 0.82]
hybrid_performance = [0.85, 0.86]

plt.figure(figsize=(10, 6))
plt.plot(personas, svd_performance, marker='s', label='SVD Baseline', color='gray', linestyle='--')
plt.plot(personas, hybrid_performance, marker='D', label='EduRec Hybrid', color='green', linewidth=2)

plt.title('NDCG Improvement Across Learner Personas', fontsize=12, fontweight='bold')
plt.ylabel('NDCG@10 Score')
plt.ylim(0.7, 0.9)
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('output/persona_performance.png', dpi=300)
plt.close()

print("All visualizations generated successfully.")
