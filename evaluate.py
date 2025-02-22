import os

import numpy as np

if __name__ == "__main__":
    result_dir = "tasks-final/"
    tasks = os.listdir(result_dir)
    total_auc = 0
    count = 0
    for task in tasks:
        if task == ".DS_Store":
            continue
        with open(result_dir + task + "/metrics.txt", "r") as f:
            data = f.readlines()
            auc = float(data[0].split()[-1])
            print(f"AUC on test set of {task}: {auc:.3f}")
            count += 1
            total_auc += auc

    print(f"Average AUC: {total_auc / len(tasks):.3f}")
