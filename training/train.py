import os, os.path as osp
import math
import argparse

import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import datasets, transforms
import horovod.torch as hvd
import tensorboardX
from tqdm import tqdm
from ..utils import * 
import horovod.torch as hvd

from pretrained_darts import NetworkCIFAR, NetworkImageNet

model_names = ["201", "DARTS", "ProxylessNAS"]
# Training settings
parser = argparse.ArgumentParser(description='PyTorch ImageNet Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('--num-classes', type=int, default=1000,
                    help='The number of classes in the dataset.')

parser.add_argument('--train-dir', default=os.path.expanduser('/ssd/dataset/imagenet/train'),
                    help='path to training data')
parser.add_argument('--val-dir', default=os.path.expanduser('/ssd/dataset/imagenet/val'),
                    help='path to validation data')
parser.add_argument('--log-dir', default='./logs',
                    help='tensorboard log directory')
parser.add_argument('--format', default='./checkpoint-{epoch}.pth.tar',
                    help='checkpoint file format')
parser.add_argument('--batch-size', type=int, default=64,
                    help='input batch size for training')
parser.add_argument('--val-batch-size', type=int, default=64,
                    help='input batch size for validation')
parser.add_argument('--epochs', type=int, default=150,
                    help='number of epochs to train')
parser.add_argument('--base-lr', type=float, default=0.0125,
                    help='learning rate for a single GPU')
parser.add_argument('--warmup-epochs', type=float, default=5,
                    help='number of warmup epochs')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--wd', type=float, default=0.00005,
                    help='weight decay')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
parser.add_argument('--lr-scheduler', type=str, default="cosine", choices=["linear", "cosine"],
                    help='how to schedule learning rate')

args = parser.parse_args()
torch.manual_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"

class CrossEntropyLabelSmooth(nn.Module):

  def __init__(self, num_classes, epsilon):
    super(CrossEntropyLabelSmooth, self).__init__()
    self.num_classes = num_classes
    self.epsilon = epsilon
    self.logsoftmax = nn.LogSoftmax(dim=1)

  def forward(self, inputs, targets):
    log_probs = self.logsoftmax(inputs)
    targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
    targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
    loss = (-targets * log_probs).mean(0).sum()
    return loss

kwargs = {'num_workers': 5, 'pin_memory': True} if args.cuda else {}
# Training transform
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
pre_process = [
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
]
if args.color_jitter:
    pre_process += [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)]
pre_process += [
    transforms.ToTensor(),
    normalize
]

train_dataset = datasets.ImageFolder(args.train_dir,
                         transform=transforms.Compose(pre_process))
# Horovod: use DistributedSampler to partition data among workers. Manually specify
# `num_replicas=hvd.size()` and `rank=hvd.rank()`.
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs)

val_dataset = datasets.ImageFolder(args.val_dir,
                         transform=transforms.Compose([
                             transforms.Resize(256),
                             transforms.CenterCrop(224),
                             transforms.ToTensor(),
                             normalize
                         ]))
val_sampler = torch.utils.data.distributed.DistributedSampler(
    val_dataset, num_replicas=hvd.size(), rank=hvd.rank())
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size,
                                         sampler=val_sampler, **kwargs)

model = NetworkCIFAR(model=args.arch, classes=args.num_classes).to(device)
optimizer = optim.SGD(model.parameters(), lr=args.base_lr * hvd.size(),
                      momentum=args.momentum, weight_decay=args.wd)


optimizer = hvd.DistributedOptimizer(optimizer,
                                     named_parameters=model.named_parameters())

criterion = nn.CrossEntropyLabelSmooth().to(device) 
criterion2 = compute_pr_metric()

## PHASE 1: Training with Strict Fairness 

epochs_p1 = 20 #PHASE 1 epochs 
for epoch in range(epochs_p1):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs, logits_aux = model(inputs.to(device))
        loss = criterion(outputs, labels.to(device))
        loss.backward()
        optimizer.step()
        if args.auxiliary:
            loss_aux = criterion(logits_aux, outputs)
            loss += args.auxiliary_weight*loss_aux


        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

acc=correct/total

# Phase 2: Pareto-aware Training 
epochs_p2 = 100
for epoch in range(epochs_p2):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs.to(device))
        loss = criterion(outputs, labels.to(device)) + criterion2(outputs, [outputs, labels.to(device)])
        loss.backward()
        optimizer.step()

        model.prune() # Remove furthest from Pareto front

        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0


    correct = 0
    total = 0
    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    