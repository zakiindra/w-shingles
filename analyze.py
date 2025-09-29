import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

LOG_DIR = 'logs'
CITY_TIMING = os.path.join(LOG_DIR, 'city_timings.log')
W_L_TIMING = os.path.join(LOG_DIR, 'shingles_and_lambda_timing.log' )

city_data = pd.read_csv(CITY_TIMING)
w_l_data = pd.read_csv(W_L_TIMING)

city_data.to_csv("results/city_timings.csv", sep=',')
w_l_data.to_csv("results/shingles_and_lambda_timing.csv", sep=",")

city_df = pd.read_csv("results/city_timings.csv")
w_l_df = pd.read_csv("results/shingles_and_lambda_timing.csv")

# Aggregate sum of total times over all cities for each (w, λ)
agg = w_l_df.groupby(["w", "lambda"])["mean_total_time"].sum().reset_index()

# Prepare for grouped bar plot
lambda_vals = sorted(agg['lambda'].unique())
w_vals = sorted(agg['w'].unique())
bar_width = 0.35
x = np.arange(len(lambda_vals))  # positions for λ

plt.figure(figsize=(10, 6))

# Plot bars for each w value
for i, w in enumerate(w_vals):
    subset = agg[agg['w'] == w]
    times = subset.set_index('lambda').reindex(lambda_vals)['mean_total_time'].values
    plt.bar(x + i*bar_width, times, width=bar_width, label=f'w={w}')

# X-axis labels in the middle of each bar pair
plt.xticks(x + bar_width/2, [str(lam) if lam != -1 else '∞' for lam in lambda_vals], rotation=45)
plt.xlabel("λ value")
plt.ylabel("Total Shingle Calculation Time (seconds)")
plt.title("Total Shingle Generation Time Across Corpus (Grouped by w)")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()

# Save and show
plt.savefig("results/shingle_timing_barplot.png", dpi=150)
plt.show()