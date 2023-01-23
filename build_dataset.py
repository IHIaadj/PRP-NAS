import os 
import numpy as np
import csv
from collections import OrderedDict
from utils import * 
import torch 
import torch.dataset as dataset
from torch.utils.data import DataLoader
import argparse


parser = argparse.ArgumentParser(description='Dataset description',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--num_classes', type=int, default=1000,
                    help='The number of classes in the dataset.')
parser.add_argument('--log-dir', default='./logs',
                    help='tensorboard log directory')
parser.add_argument('--format', default='./checkpoint-{epoch}.pth.tar',
                    help='checkpoint file format')
parser.add_argument('--batch_size', type=int, default=64,
                    help='input batch size for training')
args = parser.parse_args()

class Dataset:
    def __init__(self, folder_name, classes=args.num_classes) -> None:
        self.folder = folder_name
        self.classes = classes 
        self.data_file = "logs/data.csv"

    def split_data(self, p):
        data = Dataset(self.folder_name)
        for i in range(p):
            subset_indices = np.random.choice(data.shape[0], args.batch_size, replace=False)
            subset = torch.utils.data.Subset(data, subset_indices) 
            loader = DataLoader(
                        subset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=0,
                        collate_fn=None,
                        pin_memory=False,
                    )
            rank = is_pareto_efficient(loader)
            with open(self.data_file, "a") as f: 
                f.write(f"{loader.id},{rank}")


def main():
    new_dataset = Dataset("./data", 10)
    new_dataset.split_data()
    
if __name__ == '__main__':
    main()