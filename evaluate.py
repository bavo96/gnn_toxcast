import os

import numpy as np

if __name__ == "__main__":
    tasks = os.listdir("tasks")
    total_auc = 0
    count = 0
    for task in tasks:
        if task == ".DS_Store":
            continue
        with open("tasks/" + task + "/metrics.txt", "r") as f:
            data = f.readlines()
            auc = float(data[0].split()[-1])
            print(auc)
            count += 1
            total_auc += auc

    print("Average AUC: ", total_auc / len(tasks))
