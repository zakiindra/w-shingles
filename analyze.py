import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# --- 1. Setup Directories and File Paths ---
LOG_DIR = 'logs'
PLOT_DIR = 'plots'

# Create the plots directory if it doesn't exist
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)
    print(f"Created directory: '{PLOT_DIR}'")

# Define full paths for input and output files
W_L_TIMING_LOG = os.path.join(LOG_DIR, 'shingles_and_lambda_timing.log')
W_L_TIMING_CSV = os.path.join(LOG_DIR, 'shingles_and_lambda_timing.csv')

# --- 2. Load, Convert, and Read Data ---
try:
    # Read the primary log file
    w_l_data = pd.read_csv(W_L_TIMING_LOG)

    # For consistency and to match the original request, convert to CSV
    w_l_data.to_csv(W_L_TIMING_CSV, index=False)

    # Read the data from the newly created CSV file
    w_l_df = pd.read_csv(W_L_TIMING_CSV)
    print("Successfully loaded and processed timing data.")

except FileNotFoundError:
    print(f"Error: The required log file was not found at '{W_L_TIMING_LOG}'. Please ensure it exists.", file=sys.stderr)
    sys.exit(1) # Exit the script if the data file is missing
except Exception as e:
    print(f"An error occurred during file processing: {e}", file=sys.stderr)
    sys.exit(1)


# --- 3. Generate Plot 1: Shingle Generation Time (Bar Plot) ---
print("Generating Plot 1: Shingle Generation Time...")

# Aggregate sum of total shingle generation times for each (w, λ) pair
agg_shingle_time = w_l_df.groupby(["w", "lambda"])["mean_total_time"].sum().reset_index()

# Prepare data for the grouped bar plot
lambda_vals = sorted(agg_shingle_time['lambda'].unique())
w_vals = sorted(agg_shingle_time['w'].unique())
bar_width = 0.35
x = np.arange(len(lambda_vals))

# Create the plot figure
plt.figure(figsize=(12, 7))

# Plot a set of bars for each 'w' value
for i, w in enumerate(w_vals):
    subset = agg_shingle_time[agg_shingle_time['w'] == w]
    # Reindex to ensure times align with the full list of lambdas, filling missing with 0
    times = subset.set_index('lambda').reindex(lambda_vals)['mean_total_time'].fillna(0).values
    plt.bar(x + i * bar_width, times, width=bar_width, label=f'w={w}')

# Configure plot aesthetics
plt.xticks(x + bar_width / (len(w_vals) -1), [str(lam) if lam != -1 else '∞' for lam in lambda_vals], rotation=45)
plt.xlabel("λ (lambda) value")
plt.ylabel("Total Shingle Generation Time (seconds)")
plt.title("Total Shingle Generation Time Across Corpus (Grouped by Shingle Size w)")
plt.legend(title='Shingle Size (w)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the plot to the plots directory
plot1_path = os.path.join(PLOT_DIR, "shingle_generation_timing_barplot.png")
plt.savefig(plot1_path, dpi=150)
print(f"Plot 1 successfully saved to '{plot1_path}'")
plt.close() # Close the figure to free up memory


# --- 4. Generate Plot 2: Similarity Calculation Time (Line Plot) ---
print("\nGenerating Plot 2: Similarity Calculation Time...")

# Aggregate sum of similarity calculation times for each (w, λ) pair
corpus_time_df = w_l_df.groupby(['w', 'lambda'])['mean_similarity_time'].sum().reset_index()

# Sort values correctly for plotting, ensuring λ = -1 (∞) is last
corpus_time_df['lambda_sortable'] = corpus_time_df['lambda'].replace(-1, np.inf)
corpus_time_df = corpus_time_df.sort_values(by=['w', 'lambda_sortable'])

# Setup plotting style and figure
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 6))

# Plot a line for each 'w' value
for w_val in sorted(corpus_time_df['w'].unique()):
    w_df = corpus_time_df[corpus_time_df['w'] == w_val]
    # Create string labels for the x-axis, replacing -1 with '∞'
    x_labels = w_df['lambda'].astype(str).replace('-1', '∞')
    ax.plot(x_labels, w_df['mean_similarity_time'], marker='o', linestyle='-', label=f'w={w_val}')

# Configure plot aesthetics
ax.set_xlabel('λ (lambda) value')
ax.set_ylabel('Total Similarity Calculation Time (seconds)')
ax.set_title('Similarity Calculation Time vs. λ for Different Shingle Sizes (w)')
ax.set_yscale('log') # Use a logarithmic scale to better visualize the wide range of values
ax.legend(title='Shingle Size (w)')
plt.tight_layout()

# Save the plot to the plots directory
plot2_path = os.path.join(PLOT_DIR, "similarity_time_vs_lambda_log_scale.png")
plt.savefig(plot2_path, dpi=150)
print(f"Plot 2 successfully saved to '{plot2_path}'")
plt.close() # Close the figure

print("\nAll plots have been generated and saved.")