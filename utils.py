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
    Find the pareto-efficient points
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