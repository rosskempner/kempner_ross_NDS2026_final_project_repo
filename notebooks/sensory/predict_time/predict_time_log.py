import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.utils import resample
import os
import sys
script_path = os.path.abspath("../../../../Library/detrending_zscoring/")
if script_path not in sys.path:
    sys.path.append(script_path)
import detrending_zscoring
# Set plotting style
sns.set_theme(style="whitegrid")

# ==========================================
# 1. Generate / Load Data (Done ONCE outside the loop)
# ==========================================
gaussian_sigma = 25
# Read CSV
behavior_df = pd.read_csv(f"/home/ross/Documents/lab_projects/s1dz_sociability/data/tmp/sensory/x_y_from_crude_method_gaussianSigma{gaussian_sigma}.csv")
read_in_neural = True
if read_in_neural:
    neural_df = pd.read_csv("/home/ross/Documents/lab_projects/s1dz_sociability/data/Sensory S1DZ neurons/sensory_trial1_deconvolved.csv")
else:
    neural_df = pd.read_csv("/home/ross/Documents/lab_projects/s1dz_sociability/data/Sensory S1DZ neurons/Trial #1_Calcium signals.csv")
    import pdb
    pdb.set_trace()
    neural_df = neural_df.iloc[1:, 1:].astype(float)
    neural_df = neural_df.iloc[:len(behavior_df), :]
    # Extract base arrays
    neural_df = neural_df.apply(lambda col:\
        detrending_zscoring.correct_photobleaching_df_plus_deconvolve(col.values))
    neural_df.to_csv("/home/ross/Documents/lab_projects/s1dz_sociability/data/Sensory S1DZ neurons/sensory_trial1_deconvolved.csv", index = False)



# Extract base arrays
raw_neural_data = neural_df.to_numpy(dtype=float)
n_time_steps = raw_neural_data.shape[0]
x_coords = behavior_df["x"].to_numpy()
y_coords = behavior_df["y"].to_numpy()

# Compute Kinematics
dx = np.diff(x_coords, prepend=x_coords[0])
dy = np.diff(y_coords, prepend=y_coords[0])
velocity = np.sqrt(dx**2 + dy**2)
is_moving = np.where(velocity > 0, 1, 0)

# Zone Labeling
is_in_social_area = np.full(n_time_steps, -1)
is_in_social_area[x_coords < 220] = 1
is_in_social_area[x_coords > 450] = 0

near_cage = np.full(n_time_steps, 0)
near_cage[y_coords > 260] = 1

# Create Master DataFrame for filtering
df = pd.DataFrame({
    'original_idx': np.arange(n_time_steps),
    'is_in_social_area': is_in_social_area,
    'is_moving': is_moving,
    'velocity': velocity,
    'y_coords': y_coords,
    'near_cage': near_cage
})

moving_velocities = df[df['velocity'] > 0]['velocity']
high_vel_thresh = np.percentile(moving_velocities, 50) if not moving_velocities.empty else 0

print(f"\nCalculated High Velocity Threshold: {high_vel_thresh:.4f}")

# Filtering: Remove transitions, keep only stationary, filter by Y-coord
df_filtered = df[(df['is_in_social_area'] == 1) & (df['near_cage'] == 1) &
                  ((df['velocity'] == 0) | (df['velocity'] >= high_vel_thresh))].copy()

# ==========================================
median_trial_frame = int(len(df) / 2)

# 0 = before median, 1 = after median
df_filtered["before_after_median"] = (df_filtered["original_idx"] >= median_trial_frame).astype(int)
print(f"median_trial_frame {median_trial_frame}")
# Create strata for balancing (Social Zone + Movement state)
df_filtered['strata'] = df_filtered['before_after_median'].astype(str) #+ "_" + df_filtered['is_moving'].astype(str) #+ "_" +  df_filtered['near_cage'].astype(str)
strata_filtered = df_filtered['strata'].values
print(f"{len(df)} , {len(df_filtered)}")
print(df)
# Storage for plotting at the end
all_accuracies = []
all_conditions = []

# ==========================================
# 2. Iterative Pipeline (Permutation vs Real)
# ==========================================
for permutation in [False, True]:
    print(f"\n{'='*65}\nRUNNING PIPELINE: PERMUTATION = {permutation}\n{'='*65}")
    
    # Shuffle neural data if permutation run
    neural_data = np.random.permutation(raw_neural_data) if permutation else raw_neural_data.copy()
    X_filtered_all = neural_data[df_filtered['original_idx'].values]

    # CV Configuration: Grouped by time chunks to prevent leakage
    chunk_size = 2500
    groups = np.repeat(np.arange(len(df_filtered) // chunk_size + 1), chunk_size)[:len(df_filtered)]

    n_splits = 5
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=40)
    
    cv_scores = []
    train_sizes, test_sizes = [], []

    print(f"Starting {n_splits}-Fold CV (Train & Test Downsampled)...")

    for fold, (train_idx, test_idx) in enumerate(cv.split(X_filtered_all, strata_filtered, groups)):
        # Split dataframes
        df_train = df_filtered.iloc[train_idx].copy()
        df_test = df_filtered.iloc[test_idx].copy()
        
        # --- DOWNSAMPLE TRAINING DATA ---
        min_train_size = df_train['strata'].value_counts().min()
        train_groups = []
        for s_val, group in df_train.groupby('strata'):
            train_groups.append(resample(group, replace=False, n_samples=min_train_size))
        df_train_balanced = pd.concat(train_groups).sample(frac=1)
        
        # --- DOWNSAMPLE TEST DATA ---
        min_test_size = df_test['strata'].value_counts().min()
        test_groups = []
        for s_val, group in df_test.groupby('strata'):
            test_groups.append(resample(group, replace=False, n_samples=min_test_size))
        df_test_balanced = pd.concat(test_groups).sample(frac=1)
        
        # Logging
        print(f"\n[Fold {fold+1}] Balanced Strata Counts:")
        print(f"  Train: {df_train_balanced['strata'].value_counts().to_dict()}")
        print(f"  Test : {df_test_balanced['strata'].value_counts().to_dict()}")

        # Map back to neural data
        X_train = neural_data[df_train_balanced['original_idx'].values]
        y_train = df_train_balanced['before_after_median'].values
        X_test = neural_data[df_test_balanced['original_idx'].values]
        y_test = df_test_balanced['before_after_median'].values
        
        # Train and evaluate
        log_reg = LogisticRegression(max_iter=1000, n_jobs=-1)
        log_reg.fit(X_train, y_train)
        score = log_reg.score(X_test, y_test)
        cv_scores.append(score)
        
        print(f"  Accuracy: {score:.4f}")

    print(f"\nFinal Mean Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
    
    # Store results for final plot
    condition_label = "Permuted" if permutation else "Real"
    all_accuracies.extend(cv_scores)
    all_conditions.extend([condition_label] * len(cv_scores))

# ==========================================
# 3. Visualization
# ==========================================
# Create dataframe for Seaborn
results_df = pd.DataFrame({
    'Accuracy': all_accuracies,
    'Condition': all_conditions
})

plt.figure(figsize=(8, 6))

# Bar plot comparing Real vs Permuted
sns.barplot(data=results_df, x='Condition', y='Accuracy', palette=['skyblue', 'lightcoral'], capsize=.1, errorbar='sd')
# Overlay individual points for each fold
sns.stripplot(data=results_df, x='Condition', y='Accuracy', color='black', alpha=0.6, jitter=True)

plt.axhline(0.5, color='black', linestyle='--', linewidth=2, label='Chance (0.5)')
plt.ylim(0, 1.05)
plt.title(f'VPL-S1DZ: 1st or 2nd Half of Session', fontsize=23, fontweight='bold')

# --- Updated sizes for labels, ticks, and legend ---
plt.ylabel('Test Accuracy', fontsize=25)
plt.xlabel('')
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.legend(loc='lower right', fontsize=25)
# ---------------------------------------------------

plt.tight_layout()
out_name = "decoding_accuracy_comparison.png"
plt.savefig(out_name, dpi=300)
print(f"\nResults saved to: {os.path.abspath(out_name)}")