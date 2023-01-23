from numpy.random import rand
from numpy.random import seed
from scipy.stats import kendalltau
from scipy.stats import spearmanr
import numpy as np 
ALPHA = 0.05 

'''
Compute the kendal correlation between two variables v1 & v2 
'''
def kendal_correlation(v1, v2):
    coef, p =  kendalltau(v1, v2)

    if p > ALPHA:
        print("Samples are uncorrelated (fail to reject H0)")
        return 0
    else:
        return coef 

'''
Compute the spearman correlation between two variables v1 & v2 
'''
def spearman_correlation(v1, v2):
    coef, p =  spearmanr(v1, v2)
    if p > ALPHA:
        print("Samples are uncorrelated (fail to reject H0)")
        return 0 
    else:
        return coef 

'''
Check if two variables contains ties. 
This can help us understand which one of the two rank correlation is more significant. 
Contains ties --> Spearman 
No ties --> Kendal
'''
def check_ties(v1, v2): 
    v1_set = set(v1) 
    v2_set = set(v2) 
    if len(v1_set.intersection(v2_set)) > 0: 
        return(True)  
    return(False)    

def is_pareto_efficient(costs, return_mask = True):
    """
    Find the pareto-efficient points in a small sampled dataset  
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(costs):
        nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask, is_efficient
    else:
        return is_efficient

def compute_pr_metric(v1, costs):
    """ 
    PR stands for Pareto Ranking.
    This metric helps measure the performance of multi-objective HW-NAS. 
    """
    is_efficient = is_pareto_efficient(costs)
    is_efficient_index = costs.index(is_efficient)

    return kendal_correlation(v1, is_efficient_index)

#############################################################################################
"""DARTS Utils"""
import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms


class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(cutout, cutout_length):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if cutout:
    train_transform.transforms.append(Cutout(cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform


def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)


def save(model, model_path):
  torch.save(model.state_dict(), model_path)


def load(model, model_path, gpu_id):
  ml = 'cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu'
  model.load_state_dict(torch.load(model_path, map_location = ml), strict=False)



def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
    x.div_(keep_prob)
    x.mul_(mask)
  return x


def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)

