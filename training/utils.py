from numpy.random import rand
from numpy.random import seed
from scipy.stats import kendalltau
from scipy.stats import spearmanr
import numpy as np 
import torch
import copy
import sys
import time 
import pickle
import numpy as np
import warnings
from scipy.interpolate import Rbf

from collections import OrderedDict

from training.keys import *
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

############################################################################################################

def update_progress(index, length, **kwargs):
    '''
        display progress
        
        Input:
            `index`: (int) shows the index of current progress
            `length`: (int) total length of the progress
            `**kwargs`: info to display (e.g. accuracy)
    '''
    barLength = 10 # Modify this to change the length of the progress bar
    progress = float(index/length)
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress >= 1:
        progress = 1
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1:.2f}% ({2}/{3}) ".format( 
            "#"*block + "-"*(barLength-block), round(progress*100, 3), \
            index, length)
    for key, value in kwargs.items():
        text = text + str(key) + ': ' + str(value) + ', '
    if len(kwargs) != 0:
        text = text[:-2:]
    sys.stdout.write(text)
    sys.stdout.flush()


def get_layer_by_param_name(model, param_name):
    '''
        Get a certain layer (e.g. torch.Conv2d) from a model
        by layer parameter name (e.g. models.conv_layers.0.weight)
        
        Input: 
            `model`: model we want to get a certain layer from
            `param_name`: (string) layer parameter name
            
        Output: 
            `layer`: (e.g. torch.nn.Conv2d)
    '''
    # Get layer from model using layer name.
    layer_name_str_split = param_name.split(STRING_SEPARATOR)[:-1]
    layer = model
    for s in layer_name_str_split:
        layer = getattr(layer, s)
    return layer


def get_keys_from_ordered_dict(ordered_dict):
    '''
        get ordered list of keys from ordered dict
        
        Input: 
            `ordered_dict`
            
        Output:
            `dict_keys`
    '''
    dict_keys = []
    for key, _ in ordered_dict.items():
        dict_keys.append(key)  # get key from (key, value) pair
    return dict_keys


def extract_feature_map_sizes(model, input_data_shape):
    '''
        get conv and fc layerwise feature map size
        
        Input:
            `model`: model which we want to get layerwise feature map size.
            `input_data_shape`: (list) [C, H, W].
        
        Output:
            `fmap_sizes_dict`: (dict) layerwise feature map sizes.
        
    '''
    fmap_sizes_dict = {}
    hooks = []
    model = model.cuda()
    model.eval()

    def _register_hook(module):
        def _hook(module, input, output):
            type_str = module.__class__.__name__
            if type_str in (CONV_LAYER_TYPES + FC_LAYER_TYPES):
                module_id = id(module)
                in_fmap_size = list(input[0].size())
                out_fmap_size = list(output.size())
                fmap_sizes_dict[module_id] = {KEY_INPUT_FEATURE_MAP_SIZE: in_fmap_size,
                                              KEY_OUTPUT_FEATURE_MAP_SIZE: out_fmap_size}

        if (not isinstance(module, torch.nn.Sequential) and not isinstance(module, torch.nn.ModuleList) and not (
                module == model)):
            hooks.append(module.register_forward_hook(_hook))

    model.apply(_register_hook)
    _ = model(torch.randn([1, *input_data_shape]).cuda())
    for hook in hooks:
        hook.remove()

    return fmap_sizes_dict


def get_network_def_from_model(model, input_data_shape):
    '''
        return network def (OrderedDict) of the input model
        
        network_def only contains information about FC, Conv2d, ConvTranspose2d
        not includes batchnorm ...
  
        Input: 
            `model`: model we want to get network_def from
            `input_data_shape`: (list) [C, H, W].
        
        Output:
            `network_def`: (OrderedDict)
                           keys(): layer name (e.g. model.0.1, feature.2 ...)
                           values(): layer properties (dict)
    '''
    network_def = OrderedDict()
    state_dict = model.state_dict()

    # extract model keys in ordered manner from model dict.
    state_dict_keys = get_keys_from_ordered_dict(state_dict)

    # extract the feature map sizes.
    fmap_sizes_dict = extract_feature_map_sizes(model, input_data_shape)
    
    # for pixel shuffle
    previous_layer_name_str = None
    previous_out_channels = None
    before_squared_pixel_shuffle_factor = int(1)

    for layer_param_name in state_dict_keys:
        layer = get_layer_by_param_name(model, layer_param_name)
        layer_id = id(layer)
        layer_name_str = STRING_SEPARATOR.join(layer_param_name.split(STRING_SEPARATOR)[:-1])
        layer_type_str = layer.__class__.__name__

        # If conv layer, populate network definition.
        # WARNING: ignores maxpool and upsampling layers.
        if layer_type_str in (CONV_LAYER_TYPES + FC_LAYER_TYPES) and WEIGHTSTRING in layer_param_name:

            # Populate network def.
            if layer_type_str in FC_LAYER_TYPES:

                network_def[layer_name_str] = {
                    KEY_IS_DEPTHWISE: False,
                    KEY_NUM_IN_CHANNELS: layer.in_features,
                    KEY_NUM_OUT_CHANNELS: layer.out_features,
                    KEY_KERNEL_SIZE: (1, 1),
                    KEY_STRIDE: (1, 1),
                    KEY_PADDING: (0, 0),
                    KEY_GROUPS: 1,
                    KEY_INPUT_FEATURE_MAP_SIZE: [1, fmap_sizes_dict[layer_id][KEY_INPUT_FEATURE_MAP_SIZE][1], 1, 1],
                    KEY_OUTPUT_FEATURE_MAP_SIZE: [1, fmap_sizes_dict[layer_id][KEY_OUTPUT_FEATURE_MAP_SIZE][1], 1, 1]
                }
            else: # this means layer_type_str is in CONV_LAYER_TYPES

                # Note: Need to handle the special case when there is only one filter in the depth-wise layer
                #       because the number of groups will also be 1, which is the same as that of the point-wise layer.
                if layer.groups == 1:
                    is_depthwise = False
                else:
                    is_depthwise = True

                network_def[layer_name_str] = {
                    KEY_IS_DEPTHWISE: is_depthwise,
                    KEY_NUM_IN_CHANNELS: layer.in_channels,
                    KEY_NUM_OUT_CHANNELS: layer.out_channels,
                    KEY_KERNEL_SIZE: layer.kernel_size,
                    KEY_STRIDE: layer.stride,
                    KEY_PADDING: layer.padding,
                    KEY_GROUPS: layer.groups,
                    
                    # (1, C, H, W)
                    KEY_INPUT_FEATURE_MAP_SIZE: fmap_sizes_dict[layer_id][KEY_INPUT_FEATURE_MAP_SIZE],
                    KEY_OUTPUT_FEATURE_MAP_SIZE: fmap_sizes_dict[layer_id][KEY_OUTPUT_FEATURE_MAP_SIZE]
                }
            network_def[layer_name_str][KEY_LAYER_TYPE_STR] = layer_type_str
            

    # Support pixel shuffle.
            if layer_type_str in FC_LAYER_TYPES:
                before_squared_pixel_shuffle_factor = int(1)
            else:
                if previous_out_channels is None:
                    before_squared_pixel_shuffle_factor = int(1)
                else:
                    if previous_out_channels % layer.in_channels != 0:
                        raise ValueError('previous_out_channels is not divisible by layer.in_channels.')
                    before_squared_pixel_shuffle_factor = int(previous_out_channels / layer.in_channels)
                previous_out_channels = layer.out_channels
            if previous_layer_name_str is not None:
                network_def[previous_layer_name_str][
                    KEY_AFTER_SQUSRED_PIXEL_SHUFFLE_FACTOR] = before_squared_pixel_shuffle_factor
            network_def[layer_name_str][KEY_BEFORE_SQUARED_PIXEL_SHUFFLE_FACTOR] = before_squared_pixel_shuffle_factor
            previous_layer_name_str = layer_name_str
    if previous_layer_name_str:
        network_def[previous_layer_name_str][
        KEY_AFTER_SQUSRED_PIXEL_SHUFFLE_FACTOR] = before_squared_pixel_shuffle_factor

    return network_def


def compute_weights_and_macs(network_def):
    '''
        Compute the number of weights and MACs of a whole network.
        
        Input: 
            `network_def`: defined in get_network_def_from_model()
        
        Output:
            `layer_weights_dict`: (OrderedDict) records layerwise num of weights.
            `total_num_weights`: (int) total num of weights. 
            `layer_macs_dict`: (OrderedDict) recordes layerwise num of MACs.
            `total_num_macs`: (int) total num of MACs.     
    '''
    total_num_weights, total_num_macs = 0, 0

    # Extract conv layer names from ordered network dict.
    network_def_keys = get_keys_from_ordered_dict(network_def)

    # Init dict to store num resources for each layer.
    layer_weights_dict = OrderedDict()
    layer_macs_dict = OrderedDict()

    # Iterate over conv layers in network def.
    for layer_name in network_def_keys:
        # Take product of filter size dimensions to get num weights for layer.
        layer_num_weights = (network_def[layer_name][KEY_NUM_OUT_CHANNELS] / \
                             network_def[layer_name][KEY_GROUPS]) * \
                            network_def[layer_name][KEY_NUM_IN_CHANNELS] * \
                            network_def[layer_name][KEY_KERNEL_SIZE][0] * \
                            network_def[layer_name][KEY_KERNEL_SIZE][1]

        # Store num weights in layer dict and add to total.
        layer_weights_dict[layer_name] = layer_num_weights
        total_num_weights += layer_num_weights
        
        # Determine num macs for layer using output size.
        output_size = network_def[layer_name][KEY_OUTPUT_FEATURE_MAP_SIZE]
        output_height, output_width = output_size[2], output_size[3]
        layer_num_macs = layer_num_weights * output_width * output_height

        # Store num macs in layer dict and add to total.
        layer_macs_dict[layer_name] = layer_num_macs
        total_num_macs += layer_num_macs

    return layer_weights_dict, total_num_weights, layer_macs_dict, total_num_macs


def measure_latency(model, input_data_shape, runtimes=500):
    '''
        Measure latency of 'model'
        
        Randomly sample 'runtimes' inputs with normal distribution and
        measure the latencies
    
        Input: 
            `model`: model to be measured (e.g. torch.nn.Conv2d)
            `input_shape`: (list) input shape of the model (e.g. (B, C, H, W))
           
        Output: 
            average time (float)
    '''
    total_time = .0
    is_cuda = next(model.parameters()).is_cuda
    if is_cuda: 
        cuda_num = next(model.parameters()).get_device()
    for i in range(runtimes):       
        if is_cuda:
            input = torch.cuda.FloatTensor(*input_data_shape).normal_(0, 1)
            input = input.cuda(cuda_num)    
            with torch.no_grad():
                start = time.time()
                model(input)
                torch.cuda.synchronize()
                finish = time.time()
        else:
            input = torch.randn(input_data_shape)
            with torch.no_grad():
                start = time.time()
                model(input)
                finish = time.time()
        total_time += (finish - start)
    return total_time/float(runtimes)


def compute_latency_from_lookup_table(network_def, lookup_table_path):
    '''
        Compute the latency of all layers defined in `network_def` (only including Conv and FC).
        
        When the value of latency is not in the lookup table, that value would be interpolated.
        
        Input:
            `network_def`: defined in get_network_def_from_model()
            `lookup_table_path`: (string) path to lookup table
        
        Output: 
            `latency`: (float) latency
    '''
    latency = .0 
    with open(lookup_table_path, 'rb') as file_id:
        lookup_table = pickle.load(file_id)
    for layer_name, layer_properties in network_def.items():
        if layer_name not in lookup_table.keys():
            raise ValueError('Layer name {} in network def not found in lookup table'.format(layer_name))
            break
        num_in_channels  = layer_properties[KEY_NUM_IN_CHANNELS]
        num_out_channels = layer_properties[KEY_NUM_OUT_CHANNELS]
        if (num_in_channels, num_out_channels) in lookup_table[layer_name][KEY_LATENCY].keys():
            latency += lookup_table[layer_name][KEY_LATENCY][(num_in_channels, num_out_channels)]
        else:
            # Not found in the lookup table, then interpolate the latency
            feature_samples = np.array(list(lookup_table[layer_name][KEY_LATENCY].keys()))
            feature_samples_in  = feature_samples[:, 0]
            feature_samples_out = feature_samples[:, 1]
            measurement = np.array(list(lookup_table[layer_name][KEY_LATENCY].values()))
            assert feature_samples_in.shape == feature_samples_out.shape
            assert feature_samples_in.shape == measurement.shape
            rbf = Rbf(feature_samples_in, feature_samples_out, \
                      measurement, function='cubic')
            num_in_channels = np.array([num_in_channels])
            num_out_channels = np.array([num_out_channels])
            estimated_latency = rbf(num_in_channels, num_out_channels)
            latency += estimated_latency[0]
    return latency


def compute_resource(network_def, resource_type, lookup_table_path=None):
    '''
        compute resource based on resource type
        
        Input:
            `network_def`: defined in get_network_def_from_model()
            `resource_type`: (string) (FLOPS/WEIGHTS/LATENCY)
            `lookup_table_path`: (string) path to lookup table
        
        Output:
            `resource`: (float)
    '''
    
    if resource_type == 'FLOPS':
        _, _, _, resource = compute_weights_and_macs(network_def)
    elif resource_type == 'WEIGHTS':
        _, resource, _, _ = compute_weights_and_macs(network_def)
    elif resource_type == 'LATENCY':
        resource = compute_latency_from_lookup_table(network_def, lookup_table_path)
    else:
        raise ValueError('Only support the resource type `FLOPS`, `WEIGHTS`, and `LATENCY`.')
    return resource


def build_latency_lookup_table(model, lookup_table_path, min_conv_feature_size=8, 
                       min_fc_feature_size=128, measure_latency_batch_size=4, 
                       measure_latency_sample_times=500, verbose=False):
    '''
        Build lookup table for latencies of layers defined by `model`.
        
        Supported layers: Conv2d, Linear, ConvTranspose2d
            
        Modify get_network_def_from_model() and this function to include more layer types.
            
        input: 
            `model`: defined in get_network_def_from_model()
            `lookup_table_path`: (string) path to save the file of lookup table
            `min_conv_feature_size`: (int) The size of feature maps of simplified layers (conv layer)
                along channel dimmension are multiples of 'min_conv_feature_size'.
                The reason is that on mobile devices, the computation of (B, 7, H, W) tensors 
                would take longer time than that of (B, 8, H, W) tensors.
            `min_fc_feature_size`: (int) The size of features of simplified FC layers are 
                multiples of 'min_fc_feature_size'.
            `measure_latency_batch_size`: (int) the batch size of input data
                when running forward functions to measure latency.
            `measure_latency_sample_times`: (int) the number of times to run the forward function of 
                a layer in order to get its latency.
            `verbose`: (bool) set True to display detailed information.
    '''
    
    resource_type = 'LATENCY'
    # Generate the lookup table.
    lookup_table = OrderedDict()
    for layer_name, layer_properties in model.items():
        
        if verbose:
            print('-------------------------------------------')
            print('Measuring layer', layer_name, ':')
        
        # If the layer has the same properties as a previous layer, directly use the previous lookup table.
        for layer_name_pre, layer_properties_pre in model.items():
            if layer_name_pre == layer_name:
                break

            if layer_properties_pre == layer_properties:
                lookup_table[layer_name] = lookup_table[layer_name_pre]
                if verbose:
                    print('    Find previous layer', layer_name_pre, 'that has the same properties')
                break
        if layer_name in lookup_table:
            continue

        is_depthwise = layer_properties[KEY_IS_DEPTHWISE]
        num_in_channels = layer_properties[KEY_NUM_IN_CHANNELS]
        num_out_channels = layer_properties[KEY_NUM_OUT_CHANNELS]
        kernel_size = layer_properties[KEY_KERNEL_SIZE]
        stride = layer_properties[KEY_STRIDE]
        padding = layer_properties[KEY_PADDING]
        groups = layer_properties[KEY_GROUPS]
        layer_type_str = layer_properties[KEY_LAYER_TYPE_STR]
        input_data_shape = layer_properties[KEY_INPUT_FEATURE_MAP_SIZE]
        
        
        lookup_table[layer_name] = {}
        lookup_table[layer_name][KEY_IS_DEPTHWISE]      = is_depthwise
        lookup_table[layer_name][KEY_NUM_IN_CHANNELS]   = num_in_channels
        lookup_table[layer_name][KEY_NUM_OUT_CHANNELS]  = num_out_channels
        lookup_table[layer_name][KEY_KERNEL_SIZE]       = kernel_size
        lookup_table[layer_name][KEY_STRIDE]            = stride
        lookup_table[layer_name][KEY_PADDING]           = padding
        lookup_table[layer_name][KEY_GROUPS]            = groups
        lookup_table[layer_name][KEY_LAYER_TYPE_STR]    = layer_type_str
        lookup_table[layer_name][KEY_INPUT_FEATURE_MAP_SIZE] = input_data_shape
        lookup_table[layer_name][KEY_LATENCY]           = {}
        
        print('Is depthwise:', is_depthwise)
        print('Num in channels:', num_in_channels)
        print('Num out channels:', num_out_channels)
        print('Kernel size:', kernel_size)
        print('Stride:', stride)
        print('Padding:', padding)
        print('Groups:', groups)
        print('Input feature map size:', input_data_shape)
        print('Layer type:', layer_type_str)
        
        '''
        if num_in_channels >= min_feature_size and \
            (num_in_channels % min_feature_size != 0 or num_out_channels % min_feature_size != 0):
            raise ValueError('The number of channels is not divisible by {}.'.format(str(min_feature_size)))
        '''
        
        if layer_type_str in CONV_LAYER_TYPES:
            min_feature_size = min_conv_feature_size
        elif layer_type_str in FC_LAYER_TYPES:
            min_feature_size = min_fc_feature_size
        else:
            raise ValueError('Layer type {} not supported'.format(layer_type_str))
        
        for reduced_num_in_channels in range(num_in_channels, 0, -min_feature_size):
            if verbose:
                index = 1
                print('    Start measuring num_in_channels =', reduced_num_in_channels)
            
            if is_depthwise:
                reduced_num_out_channels_list = [reduced_num_in_channels]
            else:
                reduced_num_out_channels_list = list(range(num_out_channels, 0, -min_feature_size))
                
            for reduced_num_out_channels in reduced_num_out_channels_list:                
                if resource_type == 'LATENCY':
                    if layer_type_str == 'Conv2d':
                        if is_depthwise:
                            layer_test = torch.nn.Conv2d(reduced_num_in_channels, reduced_num_out_channels, \
                            kernel_size, stride, padding, groups=reduced_num_in_channels)
                        else:
                            layer_test = torch.nn.Conv2d(reduced_num_in_channels, reduced_num_out_channels, \
                            kernel_size, stride, padding, groups=groups)
                        input_data_shape = layer_properties[KEY_INPUT_FEATURE_MAP_SIZE]
                        input_data_shape = (measure_latency_batch_size, 
                            reduced_num_in_channels, *input_data_shape[2::])
                    elif layer_type_str == 'Linear':
                        layer_test = torch.nn.Linear(reduced_num_in_channels, reduced_num_out_channels)
                        input_data_shape = (measure_latency_batch_size, reduced_num_in_channels)
                    elif layer_type_str == 'ConvTranspose2d':
                        if is_depthwise:
                            layer_test = torch.nn.ConvTranspose2d(reduced_num_in_channels, reduced_num_out_channels, 
                                kernel_size, stride, padding, groups=reduced_num_in_channels)
                        else:
                            layer_test = torch.nn.ConvTranspose2d(reduced_num_in_channels, reduced_num_out_channels, 
                                kernel_size, stride, padding, groups=groups)
                        input_data_shape = layer_properties[KEY_INPUT_FEATURE_MAP_SIZE]
                        input_data_shape = (measure_latency_batch_size, 
                            reduced_num_in_channels, *input_data_shape[2::])
                    else:
                        raise ValueError('Not support this type of layer.')
                    if torch.cuda.is_available():
                        layer_test = layer_test.cuda()
                    measurement = measure_latency(layer_test, input_data_shape, measure_latency_sample_times)
                else:
                    raise ValueError('Only support building the lookup table for `LATENCY`.')


                # Add the measurement into the lookup table.
                lookup_table[layer_name][KEY_LATENCY][(reduced_num_in_channels, reduced_num_out_channels)] = measurement
                
                if verbose:
                    update_progress(index, len(reduced_num_out_channels_list), latency=str(measurement))
                    index = index + 1
                    
            if verbose:
                print(' ')
                print('    Finish measuring num_in_channels =', reduced_num_in_channels)
    # Save the lookup table.
    with open(lookup_table_path, 'wb') as file_id:
        pickle.dump(lookup_table, file_id)      
    return 

