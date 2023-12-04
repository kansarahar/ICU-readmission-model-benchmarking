

import matplotlib.pyplot as plt
import pandas as pd

#***********************************************************************************
# RNN Concatenated Delta Time Model Metrics per Epochs Visualization
#***********************************************************************************

df = pd.read_csv('./data/results/rnn_concat_time_deltatraining_results.csv')
df['epoch'] = [i for i in range(1, 21)]

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(df['epoch'], df['accuracy_scores'], label='Accuracy', marker='o')
ax.plot(df['epoch'], df['precision_scores'], label='Precision', marker='s')
ax.plot(df['epoch'], df['roc_auc_scores'], label='ROC AUC', marker='^')
ax.plot(df['epoch'], df['f1_scores'], label='F1 Score', marker='d')

ax.set_xlabel('Epoch')
ax.set_ylabel('Scores')
ax.legend()
ax.set_title('rnn_concat_time_deltatraining_results')
plt.show()



#***********************************************************************************
# ODE RNN Model Metrics per Epochs Visualization
#***********************************************************************************

df = pd.read_csv('./data/results/ode_rnntraining_results.csv')
df['epoch'] = [i for i in range(1, 21)]

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(df['epoch'], df['accuracy_scores'], label='Accuracy', marker='o')
ax.plot(df['epoch'], df['precision_scores'], label='Precision', marker='s')
ax.plot(df['epoch'], df['roc_auc_scores'], label='ROC AUC', marker='^')
ax.plot(df['epoch'], df['f1_scores'], label='F1 Score', marker='d')

ax.set_xlabel('Epoch')
ax.set_ylabel('Scores')
ax.legend()
ax.set_title('ODE_RNN')
plt.show()


#***********************************************************************************
# RNN Exponential Decay Model Metrics per Epochs Visualization
#***********************************************************************************

df = pd.read_csv('./data/results/rnn_exp_decaytraining_results.csv')
df['epoch'] = [i for i in range(1, 6)]

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(df['epoch'], df['accuracy_scores'], label='Accuracy', marker='o')
ax.plot(df['epoch'], df['precision_scores'], label='Precision', marker='s')
ax.plot(df['epoch'], df['roc_auc_scores'], label='ROC AUC', marker='^')
ax.plot(df['epoch'], df['f1_scores'], label='F1 Score', marker='d')

ax.set_xlabel('Epoch')
ax.set_ylabel('Scores')
ax.legend()
ax.set_title('rnn_exp_decaytraining_results')
plt.show()


#***********************************************************************************
# Visualizing Comparison of Metrics between Validation and Test
#***********************************************************************************

# Load the results for each model
ode_rnn_results = pd.read_csv('./data/results/ode_rnntraining_results.csv')
rnn_exp_decay_results = pd.read_csv('./data/results/rnn_exp_decaytraining_results.csv')
rnn_concat_time_delta_results = pd.read_csv('./data/results/rnn_concat_time_deltatraining_results.csv')

# Calculate the mean of each metric for each model
ode_rnn_mean = ode_rnn_results.mean()
rnn_exp_decay_mean = rnn_exp_decay_results.mean()
rnn_concat_time_delta_mean = rnn_concat_time_delta_results.mean()

# Create a summary table with the means
summary_table = pd.DataFrame({
    'Model': ['ODE_RNN', 'RNN_EXP_Decay', 'RNN_Concat_Time_Delta'],
   #  'Training Loss': [ode_rnn_mean['training_losses'], rnn_exp_decay_mean['training_losses'], rnn_concat_time_delta_mean['training_losses']],
    'Validation Loss': [ode_rnn_mean['validation_losses'], rnn_exp_decay_mean['validation_losses'], rnn_concat_time_delta_mean['validation_losses']],
    'Accuracy': [ode_rnn_mean['accuracy_scores'], rnn_exp_decay_mean['accuracy_scores'], rnn_concat_time_delta_mean['accuracy_scores']],
    'Precision': [ode_rnn_mean['precision_scores'], rnn_exp_decay_mean['precision_scores'], rnn_concat_time_delta_mean['precision_scores']],
    'ROC AUC': [ode_rnn_mean['roc_auc_scores'], rnn_exp_decay_mean['roc_auc_scores'], rnn_concat_time_delta_mean['roc_auc_scores']],
    'F1 Score': [ode_rnn_mean['f1_scores'], rnn_exp_decay_mean['f1_scores'], rnn_concat_time_delta_mean['f1_scores']]
})


# Adding in Test Results as well from the runs across three models
data = {
    "Model": ["ODE_RNN", "RNN_EXP_Decay", "RNN_Concat_Time_Delta"] * 2,
    "Phase": ["Validation", "Validation", "Validation", "Testing", "Testing", "Testing"],
    "Accuracy": [0.745671, 0.682075, 0.708105, 0.680458, 0.693772, 0.636004],
    "Precision": [0.191595, 0.193928, 0.186277, 0.176105, 0.187264, 0.175744],
    "ROC AUC": [0.654859, 0.678909, 0.654729, 0.643155, 0.662314, 0.651141],
    "F1 Score": [0.337406, 0.341145, 0.330850, 0.313150, 0.332133, 0.310546],
    "Loss": [85.209088, 77.447353, 83.823950, 327.170666, 335.003268, 333.530115]
}

# Create DataFrame
df = pd.DataFrame(data)

# To display the DataFrame correctly in a more conventional format
df = df.pivot_table(index=["Model", "Phase"])
df = df.reindex(["Accuracy", "Precision", "ROC AUC", "F1 Score", "Loss"], axis=1)
df.reset_index(inplace=True)
df = df.sort_values(by=["Model", "Phase"], ascending=[True, False])


metrics = ["Accuracy", "Precision", "ROC AUC", "F1 Score"]
colors = {"ODE_RNN": "blue", "RNN_EXP_Decay": "green", "RNN_Concat_Time_Delta": "red"}
markers = {"Validation": "o", "Testing": "s"}

fig, axs = plt.subplots(1, len(metrics), figsize=(20, 5))

for i, metric in enumerate(metrics):
    for _, row in df.iterrows():
        axs[i].scatter(row["Model"], row[metric], color=colors[row["Model"]], marker=markers[row["Phase"]], label=row["Phase"])
    axs[i].set_title(metric)
    axs[i].set_xlabel("Model")
    axs[i].set_ylabel(metric)


# Removing duplicate labels from the legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
# Setting legend colors to black
for handle in by_label.values():
    handle.set_color('black')
fig.legend(by_label.values(), by_label.keys(), loc="upper center", ncol=6)

plt.tight_layout()
plt.savefig("myImagePDF.pdf", format="pdf", bbox_inches="tight")
plt.show()