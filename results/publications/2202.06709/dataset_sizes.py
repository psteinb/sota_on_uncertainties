import sys
from pathlib import Path

import ops.datasets as datasets
from torch.utils.data import DataLoader


def main(dname="cifar100"):

    dlpath = Path("./mldata")
    if not dlpath.is_dir():
        print("creating ", dlpath)
        dlpath.mkdir()
    # CIFAR100
    dataset_train, dataset_test = datasets.get_dataset(
        dname, root="./mldata", download=True
    )
    dataset_name = dname

    num_classes = len(dataset_train.classes)

    dataset_train = DataLoader(dataset_train, shuffle=True)
    dataset_test = DataLoader(dataset_test)

    print(
        "Train: %s, Test: %s, Classes: %s"
        % (len(dataset_train.dataset), len(dataset_test.dataset), num_classes)
    )


if __name__ == "__main__":
    main()
