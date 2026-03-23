import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import os

# Ensure output directory exists
output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# The Data
data = {
    'Stage': ['Registration', 'Start', 'Week 1', 'Week 2', 'Week 3', 'Week 4', 'Completion'],
    'Retention': [100, 55, 25, 15, 12, 10, 5.5]
}

df = pd.DataFrame(data)

# Create the Graph
plt.figure(figsize=(10, 6))
plt.plot(df['Stage'], df['Retention'], marker='o', linestyle='-', color='#c0392b', linewidth=3, markersize=8)

# Add title and labels
plt.title('The MOOC "L-Curve": Attrition from Registration to Certification', fontsize=14, fontweight='bold')
plt.ylabel('Percentage of Enrolled Students', fontsize=12)
plt.xlabel('Course Timeline', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

# Add annotations for context
plt.annotate('The "No-Show" Gap\n(45% never login)', xy=('Start', 55), xytext=('Start', 75),
             arrowprops=dict(facecolor='black', shrink=0.05), ha='center')

plt.annotate('Only ~5% Certify', xy=('Completion', 5.5), xytext=('Week 4', 25),
             arrowprops=dict(facecolor='black', shrink=0.05), ha='center')

# Fill area under the curve
plt.fill_between(df['Stage'], df['Retention'], color='#e74c3c', alpha=0.1)

# Set Y-axis limit
plt.ylim(0, 105)

plt.tight_layout()

# Save instead of show
output_path = os.path.join(output_dir, 'mooc_retention_curve.png')
plt.savefig(output_path)
print(f"Graph saved to {output_path}")
