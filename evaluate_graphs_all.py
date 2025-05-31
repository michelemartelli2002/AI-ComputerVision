import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.gridspec as gridspec


file_names = ["net_logs/cumulative_epoch_duration.csv",
              "net_logs/cumulative_test_acc.csv",
              "net_logs/cumulative_test_loss.csv",
              "net_logs/cumulative_train_acc.csv",
              "net_logs/cumulative_train_loss.csv"]
title = "Caltech101 Complete Benchmark"

dfs = []
for i in range(5):
    dfs.append(pd.read_csv(file_names[i]))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
net_names = ["AlexNet", "AlexNetV2", "AlexNetV3", "Pre-Trained AlexNet", "Weak AlexNet"]
column_names = dfs[0].columns.to_list()

# Create the figure
fig = plt.figure(figsize=(15, 9))

# Set up the grid: 2 rows, 2 columns
gs = gridspec.GridSpec(2, 2, figure=fig)

x = range(1, len(dfs[1]) + 1)

ax1 = fig.add_subplot(gs[0, 0])
# Plot  duration
for i in range(5):
    ax1.plot(x, dfs[0][column_names[i]], label=net_names[i], color=colors[i])

ax1.set_xlabel(xlabel="Epoch number")
ax1.set_ylabel(ylabel="(s)")
ax1.set_title(f"{title} Epoch duration")
ax1.grid(True)
ax1.legend()
ax1.set_ylim([0, dfs[0].max().max() + 2])

ax2 = fig.add_subplot(gs[0, 1])
# Plot loss
for i in range(5):
    # Train Loss
    ax2.plot(x, dfs[4][column_names[i]], label=f"{net_names[i]} Train Loss", color=colors[i])
    # Test Loss
    ax2.plot(x, dfs[2][column_names[i]], label=f"{net_names[i]} Validation Loss", color=colors[i], linestyle='--')

ax2.set_xlabel(xlabel="Epochs")
ax2.set_title(f"{title} Loss")
ax2.set_ylim([-0.1, 5])
ax2.grid(True)
ax2.legend()

ax3 = fig.add_subplot(gs[1, :])

# Plot accuracy
for i in range(5):
    # Train Acc
    ax3.plot(x, dfs[3][column_names[i]], label=f"{net_names[i]} Train Accuracy", color=colors[i])
    # Test Acc
    ax3.plot(x, dfs[1][column_names[i]], label=f"{net_names[i]} Validation Accuracy", color=colors[i], linestyle='--')

ax3.set_xlabel(xlabel="Epochs")
ax3.set_ylabel(ylabel="(%)")
ax3.set_title(f"{title} Accuracy")
ax3.grid(True)
ax3.legend()
ax3.set_ylim([0, 1.1])


plt.tight_layout()
output_file_name = "metrics/" + title.lower().replace(" ", "_") + ".pdf"
plt.savefig(output_file_name)
print(f"Saved {output_file_name}")

