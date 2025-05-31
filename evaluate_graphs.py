import matplotlib.pyplot as plt
import pandas as pd

file_name = "net_logs/alexnet_cifar10.csv"
title = "AlexNet CIFAR10"
df = pd.read_csv(file_name)

fig, axes = plt.subplots(3, 1, figsize=(6, 9))

x = range(1, len(df) + 1)
y1 = df["Train_Loss"]
y2 = df["Test_Loss"]
y3 = df["Train_Acc"]
y4 = df["Test_Acc"]
y5 = df["Epoch_Duration"]

axes[0].plot(x, y1, label="Train Loss", color='tab:blue')
axes[0].plot(x, y2, linestyle='--', label="Test Loss", color='tab:orange')
axes[0].set_xlabel(xlabel="Epochs")
axes[0].set_title(f"{title} Loss")
axes[0].grid(True)
axes[0].legend()


axes[1].plot(x, y3, label="Train Accuracy", color='tab:blue')
axes[1].plot(x, y4, linestyle='--', label="Test Accuracy", color='tab:orange')
axes[1].set_xlabel(xlabel="Epochs")
axes[1].set_ylabel(ylabel="(%)")
axes[1].set_title(f"{title} Accuracy")
axes[1].grid(True)
axes[1].legend()
axes[1].set_ylim([0, 1])


axes[2].plot(x, y5, label="Epoch duration", color='tab:green')
axes[2].set_xlabel(xlabel="Epoch number")
axes[2].set_ylabel(ylabel="(s)")
axes[2].set_title(f"{title} Epoch duration")
axes[2].grid(True)
axes[2].legend()
axes[2].set_ylim([0, y5.max() + 10])

plt.tight_layout()
output_file_name = "metrics/" + title.lower().replace(" ", "_") + ".pdf"
plt.savefig(output_file_name)
print(f"Saved {output_file_name}")

