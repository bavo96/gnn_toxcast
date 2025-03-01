import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

from data import smiles_to_data
from model import CustomGNN

np.random.seed(0)  # Set seed for reproducibility
torch.manual_seed(0)
random.seed(0)


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


if __name__ == "__main__":
    result_dir = "tasks-final/"
    df = pd.read_csv("toxcast_data.csv")
    print(f"Number of rows in original dataset: {df.shape[0]}")
    print(f"Number of tasks: {df.shape[1] - 1}")
    print(f"Some Tasks: {list(df.columns.values)[1:5]}")

    best_dist = get_data_distribution(df)
    # print(f"Number of tasks that aren't imbalanced: {len(best_dist)}")
    # print(f"Some Tasks: {best_dist[:5]}")

    print("Checking models...")
    for task in best_dist:
        print(f"===> Processing {task[0]}...")
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

        # print(f"Number of rows after preprocessing: {len(data_list)}")

        # (Optional) Split into train and test sets
        batch_size = 32
        train_data, test_data = train_test_split(
            data_list, test_size=0.2, random_state=42, stratify=labels
        )
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

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
        model.load_state_dict(
            torch.load(
                os.path.join(result_dir, task[0], "best_models/best_model.pt"),
                weights_only=True,
            )
        )
        train_acc, train_auc, train_cf = test(model, train_loader)
        test_acc, test_auc, test_cf = test(model, test_loader)

        with open(result_dir + task[0] + "/metrics.txt", "r") as f:
            auc = float(f.readlines()[0].split()[-1])
            if auc != test_auc:
                print(f"Incorrect AUC: {task[0]}")
    print("All correct.")
