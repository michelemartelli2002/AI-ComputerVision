import torch
from sklearn.metrics import classification_report, confusion_matrix
import dataset.dataset as dataset
from model import alexnet, pretrained_alexnet, weak_alexnet, improved_alexnet, improved_attention_alexnet
import matplotlib.pyplot as plt
import seaborn as sns
from torchsummary import summary
import pandas as pd
import sys

def save_classification_matrix(cm, class_names, title, color, normalized):
    size = max(10, int(len(class_names) / 5))
    plt.figure(figsize=(size, size))

    fmt = 'd'
    if normalized=='true':
        fmt = '.2f'

    sns.heatmap(cm,
                annot=True,
                annot_kws={"size": 3} if len(class_names) >= 20 else {},   # Print smaller numbers if there are too many classes
                fmt=fmt,
                xticklabels=class_names,
                yticklabels=class_names,
                cmap=color,
                linewidths=0.5,
                linecolor="gray")

    plt.xticks(rotation=90 if len(class_names) >= 10 else 0,
               fontsize=10 if len(class_names) >=10 else 5)
    plt.yticks(fontsize=10 if len(class_names) >=10 else 5)


    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix | {title}")
    plt.tight_layout()

    output_file_name = "metrics/cm_" + title.lower().replace(" ", "_") + ".pdf"
    plt.savefig(output_file_name)
    print(f"Saved {output_file_name}")


def evaluate_model(model, dataloader, class_names, device, dataset_name):
    model.eval()
    model.to(device)
    net_name_lowercase = model.name.lower()

    summary_file_name = f"metrics/{net_name_lowercase}_{dataset_name}_summary.txt"
    with open(summary_file_name, "w") as f:
        stdout = sys.stdout
        sys.stdout = f
        summary(model, input_size=(3, 224, 224), device=str(device))
        sys.stdout = stdout
        print(f"Saved {summary_file_name}")

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # Classification report
    report = classification_report(all_labels,
                                   all_preds,
                                   target_names=class_names,
                                   zero_division=0,
                                   digits=3)
    classification_report_file_name = f"metrics/{net_name_lowercase}_{dataset_name}_classification_report.txt"
    with open(classification_report_file_name, "w") as f:
        f.write(report + "\n")

    print(f"Saved {classification_report_file_name}")

    normalize = None

    if (dataset_name == "CIFAR10"): color = "Blues"; normalize = None
    if (dataset_name == "CIFAR100"): color = "Oranges"; normalize = None
    if (dataset_name == "Flowers102"): color = "Greens"; normalize = 'true'
    if (dataset_name == "Caltech101"): color = "Purples"; normalize = 'true'


    # Confusion matrix

    cm = confusion_matrix(all_labels, all_preds, normalize=normalize)
    save_classification_matrix(cm=cm,
                               class_names=class_names,
                               title=f"{dataset_name} {model.name}",
                               color=color,
                               normalized=normalize)

def evaluate_cifar10(device, model):
    test_loader = dataset.get_cifar10_test_loader(batch_size=64,
                                                  num_workers=6,
                                                  resize_w=224,
                                                  resize_h=224)

    class_names = dataset.get_cifar10_class_names()

    evaluate_model(model=model,
                   dataloader=test_loader,
                   class_names=class_names,
                   device=device,
                   dataset_name="CIFAR10")

def evaluate_cifar100(device, model):
    test_loader = dataset.get_cifar100_test_loader(batch_size=64,
                                                   num_workers=6,
                                                   resize_w=224,
                                                   resize_h=224)

    class_names = dataset.get_cifar100_class_names()

    evaluate_model(model=model,
                   dataloader=test_loader,
                   class_names=class_names,
                   device=device,
                   dataset_name="CIFAR100")

def evaluate_flowers102(device, model):
    test_loader = dataset.get_flowers102_test_loader(batch_size=64,
                                                     num_workers=6,
                                                     resize_w=224,
                                                     resize_h=224)

    class_names = dataset.get_flowers102_class_names()

    evaluate_model(model=model,
                   dataloader=test_loader,
                   class_names=class_names,
                   device=device,
                   dataset_name="Flowers102")

def evaluate_caltech101(device, model):
    test_loader = dataset.get_caltech101_test_loader(batch_size=64,
                                                     num_workers=6,
                                                     resize_w=224,
                                                     resize_h=224)

    class_names = dataset.get_caltech101_class_names()

    evaluate_model(model=model,
                   dataloader=test_loader,
                   class_names=class_names,
                   device=device,
                   dataset_name="Caltech101")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = pretrained_alexnet.get_caltech101_model()
    model.load_state_dict(torch.load("model/saved/pre-trained_alexnet_trained50_caltech101.pth",
                                     map_location=device))

    evaluate_caltech101(device=device,
                        model=model)

if __name__ == "__main__":
    main()
