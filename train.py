import os
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

from data import smiles_to_data
from model import CustomGNN, FocalLoss

np.random.seed(0)  # Set seed for reproducibility
torch.manual_seed(0)
random.seed(0)


def train(model, optimizer, criterion, train_loader):
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(model, test_loader):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    correct = 0
    y_pred, y_true = [], []

    for data in test_loader:
        data = data.to(device)
        out = model(data)
        pred = out.argmax(dim=1)
        y_pred.extend(pred.tolist())
        y_true.extend(data.y.view(-1).tolist())
        correct += (pred == data.y.view(-1)).sum().item()
    # print(len(y_pred), len(y_true), len(loader.dataset))
    # print(y_pred, y_true)
    auc = roc_auc_score(y_true, y_pred)
    cf = confusion_matrix(y_true, y_pred)
    return correct / len(test_loader.dataset), auc, cf


def get_data_distribution(df):
    best_dist = []
    for task in list(df.columns.values)[1:]:
        # print(f"Processing {task}...")
        # data_list = []
        zeros = df[task].value_counts(dropna=True)[0]
        ones = df[task].value_counts(dropna=True)[1]

        dist = ones / (ones + zeros)
        # print(f"Total: {ones+zeros}, ones: {ones}, ratio: {dist}")
        if dist > 0.5 and dist < 0.6:
            best_dist.append((task, dist))
    # print(f"Best task: {_label}, distribution: {best_dist}")
    return best_dist


def save_fig(cf, parent_dir, fig_name):
    disp = ConfusionMatrixDisplay(confusion_matrix=cf, display_labels=["0", "1"])
    disp.plot()
    output_path = os.path.join(parent_dir, f"{fig_name}.png")
    plt.savefig(output_path)  # Save as PNG


if __name__ == "__main__":
    df = pd.read_csv("toxcast_data.csv")
    print(f"Number of rows in original dataset: {df.shape[0]}")
    print(f"Number of tasks: {df.shape[1] - 1}")
    print(f"Some Tasks: {list(df.columns.values)[1:5]}")

    best_dist = get_data_distribution(df)
    print(f"Number of tasks that aren't imbalanced: {len(best_dist)}")
    print(f"Some Tasks: {best_dist[:5]}")

    # exception = "NCCT_QuantiLum_inhib_dn"
    for i, task in enumerate(best_dist):
        # if exception not in task[0]:
        #     continue

        print(f"===> Processing {task[0]}, {i+1}/{len(best_dist)}...")
        ### Load model
        model = CustomGNN(
            in_channels=1,
            hidden_channels=128,
            out_channels=2,
            edge_dim=4,
            num_layers=3,
            dropout=0.3,
            slices=2,
            f_att=True,
            r=4,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=5e-4)
        criterion = FocalLoss()

        ### Load data
        labels = []
        data_list = []
        for idx, row in df.iterrows():
            # print(row["ACEA_T47D_80hr_Negative"], pd.isna(row["ACEA_T47D_80hr_Negative"]))
            if not (pd.isna(row[task[0]])):
                data_obj = smiles_to_data(row["smiles"], row[task[0]])
                if data_obj is not None:
                    labels.append(data_obj.y)
                    data_list.append(data_obj)

        print(f"Number of rows after preprocessing: {len(data_list)}")

        # (Optional) Split into train and test sets
        batch_size = 32
        train_data, test_data = train_test_split(
            data_list, test_size=0.2, random_state=42, stratify=labels
        )
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        train_ones = [1 for sample in train_data if sample.y == 1]
        train_zeros = [0 for sample in train_data if sample.y == 0]
        test_ones = [1 for sample in test_data if sample.y == 1]
        test_zeros = [0 for sample in test_data if sample.y == 0]
        print(f"Number of training data: {len(train_data)}")
        print(f"    - Number of ones: {len(train_ones)}")
        print(f"    - Number of zeros: {len(train_zeros)}")
        print(f"Number of test data: {len(test_data)}")
        print(f"    - Number of ones: {len(test_ones)}")
        print(f"    - Number of zeros: {len(test_zeros)}")

        # Training loop
        num_epochs = 500
        best_auc = 0
        model_path = "best_models"
        fig_path = "figs"
        metrics_name = "metrics.txt"

        Path(os.path.join("tasks", task[0], model_path)).mkdir(
            parents=True, exist_ok=True
        )
        Path(os.path.join("tasks", task[0], fig_path)).mkdir(
            parents=True, exist_ok=True
        )
        output_file = open(os.path.join("tasks", task[0], metrics_name), "w")

        start = time.time()
        for epoch in range(1, num_epochs + 1):
            loss = train(model, optimizer, criterion, train_loader)
            train_acc, train_auc, train_cf = test(model, train_loader)
            test_acc, test_auc, test_cf = test(model, test_loader)
            if epoch % 10 == 0:
                print(f"Epoch: {epoch:3d}, Loss: {loss:.4f}")
                print(f"    - Train Acc {train_acc:.4f}, Train AUC: {train_auc:.4f}")
                print(f"    - Test Acc: {test_acc:.4f}, Test AUC: {test_auc:.4f}")
                if test_auc > best_auc:
                    best_auc = test_auc
                    save_fig(
                        train_cf, os.path.join("tasks", task[0], fig_path), "train"
                    )
                    save_fig(test_cf, os.path.join("tasks", task[0], fig_path), "test")
                    torch.save(
                        model.state_dict(),
                        os.path.join("tasks", task[0], model_path, "best_model.pt"),
                    )
        end = time.time() - start
        output_file.write(f"Best AUC on test set: {best_auc}\n")
        output_file.write(f"Training time: {end} seconds")
