import kornia
import time
import copy
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils import load_mask, l1_norm, facemask_matrix
from config import config

def utgt_pgd(mode, model, X, y, epsilon=config['pgd']['epsilon'], alpha=config['pgd']['alpha'], 
                    step=config['pgd']['step'], univ=False):
    """
    Implementation of untargeted PGD attack on face recognition.
    This attack add imperceptible noise on the whole image.
    Args:
        mode (string): the type of target system, can be one of ['closed', 'open'].
        model: the target model robustness of which is being evaluated.
        X (torch.Tensor): the input image batch with size [batch_size, 3, 224, 224].
        y (torch.Tensor): the labels of input images.
        epsilon (float): the largest l-infinity distortion of attack, should be between 0 and 1, typically we use 8/255.
        alpha (float): step size of PGD attack.
        step (int): number of attack steps.
        univ (bool): whether to produce a universal adversarial example for X. If false, then each image in X has its own perturbation;
        otherwise, the whole batch share a universal perturbation.
    Returns:
        adversarial perturbation (torch.Tensor). If univ=True, then the perturbation has the same size as X[0]; otherwise, it has the same size as X.
    """        
    
    # Initialize perturbation, randomly sampled in [-epsilon, epsilon].        
    # Make sure the initial perturbation is feasible.
    if univ:
        delta = torch.zeros_like(X[0], requires_grad=True)
        delta.data = delta.detach() * 2 * epsilon - epsilon
        ub = torch.min(1-X, dim=0)[0]
        lb = torch.max(-X, dim=0)[0]
        delta.data = torch.min(torch.max(delta.detach(), lb), ub)
    else:
        delta = torch.zeros_like(X, requires_grad=True)
        delta.data = delta.detach() * 2 * epsilon - epsilon
        delta.data = torch.min(torch.max(delta.detach(), -X), 1-X)

    cos = nn.CosineSimilarity(dim=1, eps=1e-8)

    # Iteratively update delta.
    for t in range(step):
        if univ:
            if mode == 'closed':
                loss = torch.min(nn.CrossEntropyLoss(reduction='none')(model(X+delta), y))
            else:
                loss = torch.min(1-cos(model(X), model(X+delta)))
        else:
            if mode == 'closed':
                loss = nn.CrossEntropyLoss()(model(X+delta), y)
            else:
                loss = torch.sum(1-cos(model(X), model(X+delta)))
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)

        # Note that delta should be clipped such that X+delta lies in the [0,1] range.
        # That is, delta should lies in the [-X, 1-X] range for each x in X.
        if univ:
            delta.data = torch.min(torch.max(delta.detach(), lb), ub)
        else:
            delta.data = torch.min(torch.max(delta.detach(), -X), 1-X)
        delta.grad.zero_()
                    
    return delta.detach()  

def utgt_occlusion(mode, model, X, y, mask, epsilon=config['occlusion']['epsilon'], alpha=config['occlusion']['alpha'], 
                          step=config['occlusion']['step'], univ=False):
    """
    Implementation of untargeted occlusion attack on face recognition.
    This attack add occlusion with pixel-level noise on a restricted area of human face.
    We can use this function to produce sitcker and eyeglass attacks.   
    Args:
        mode (string): the type of target system, can be one of ['closed', 'open'].
        model: the target model robustness of which is being evaluated.
        X (torch.Tensor): the input image batch with size [batch_size, 3, 224, 224].
        y (torch.Tensor): the labels of input images.
        mask (torch.Tensor): the area where occlusion is added. size: [3, 224, 224]. 
        epsilon (float): the largest l-infinity distortion of attack, should be between 0 and 1, typically we use 8/255.
        alpha (float): step size of PGD attack.
        step (int): number of attack steps.
        univ (bool): whether to produce a universal adversarial example for X. If false, then each image in X has its own perturbation.
    Returns:
        adversarial perturbation (torch.tensor) which has the same size as X, but occlusion is restricted.
    """     
    
    # Initialize the perturbation with grey color.
    if univ:
        delta = torch.ones_like(X[0], requires_grad=True)
    else:
        delta = torch.ones_like(X, requires_grad=True)
    delta.data = delta.detach()*mask*128/255.
    
    cos = nn.CosineSimilarity(dim=1, eps=1e-8)

    # Iteratively update delta.
    for t in range(step):
        if univ:
            if mode == 'closed':
                loss = torch.min(nn.CrossEntropyLoss(reduction='none')(model(X*(1-mask) + delta), y))
            else:
                loss = torch.min(1-cos(model(X), model(X*(1-mask) + delta)))
        else:
            if mode == 'closed':
                loss = nn.CrossEntropyLoss()(model(X*(1-mask) + delta), y)
            else:
                loss = torch.sum(1-cos(model(X), model(X*(1-mask) + delta)))
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.data = delta.detach()*mask
        delta.grad.zero_()
        
    return delta.detach()

def utgt_facemask(mode, model, X, y, mask, epsilon=config['facemask']['epsilon'], alpha=config['facemask']['alpha'], 
                         step=config['facemask']['step'], univ=False):
    """
    Implementation of untargeted face mask attack on face recognition.
    This attack add occlusion with grid-level perturbation on a face mask.
    Args:
        mode (string): the type of target system, can be one of ['closed', 'open'].
        model: the target model robustness of which is being evaluated.
        X (torch.Tensor): the input image batch with size [batch_size, 3, 224, 224].
        y (torch.Tensor): the labels of input images.
        mask (torch.Tensor): the area where occlusion is added. size: [3, 224, 224]. 
        epsilon (float): the largest l-infinity distortion of attack, should be between 0 and 1, typically we use 8/255.
        alpha (float): step size of PGD attack.
        step (int): number of attack steps.
        univ (bool): whether to produce a universal adversarial example for X. If false, then each image in X has its own perturbation. 
    Returns:
        adversarial perturbation (torch.tensor) which has the same size as X, but occlusion only occurs in the face mask area.
    """         

    mask_left, mask_right, T_left, T_right = facemask_matrix()
    dimension_0 = 1 if univ else X.shape[0]
    temp = torch.zeros(dimension_0, 3, config['facemask']['height'], config['facemask']['width']).cuda()

    # Initialize the face mask with grey color
    delta = torch.ones_like(temp, requires_grad=True)
    delta.data = delta.detach()*128/255.

    # From color-grids to 2-D rectangulat face mask
    delta_large = F.interpolate(delta, size=[80, 160])
    delta_large_left = delta_large*mask_left
    delta_large_right = delta_large*mask_right
    
    # From 2-D to 3-D by using double perspective transformation
    facemask_left = kornia.warp_perspective(delta_large_left*255., T_left.repeat(dimension_0, 1, 1, 1), dsize=(224, 224))/255.
    facemask_right = kornia.warp_perspective(delta_large_right*255., T_right.repeat(dimension_0, 1, 1, 1), dsize=(224, 224))/255.
    facemask = facemask_left + facemask_right
    
    g = torch.zeros_like(delta)
    cos = nn.CosineSimilarity(dim=1, eps=1e-8)

    # Iteratively update delta.
    for t in range(step):
        if univ:
            if mode == 'closed':
                loss = torch.min(nn.CrossEntropyLoss(reduction='none')(model(X*(1-mask) + facemask), y))
            else:
                loss = torch.min(1-cos(model(X), model(X*(1-mask) + facemask)))
        else:
            if mode == 'closed':
                loss = nn.CrossEntropyLoss()(model(X*(1-mask) + facemask), y)
            else:
                loss = torch.sum(1-cos(model(X), model(X*(1-mask) + facemask)))
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        
        delta_large = F.interpolate(delta, size=[80, 160])
        delta_large_left = delta_large*mask_left
        delta_large_right = delta_large*mask_right
        
        facemask_left = kornia.warp_perspective(delta_large_left*255., T_left.repeat(dimension_0, 1, 1, 1), dsize=(224, 224))/255.
        facemask_right = kornia.warp_perspective(delta_large_right*255., T_right.repeat(dimension_0, 1, 1, 1), dsize=(224, 224))/255.
        facemask = facemask_left + facemask_right
        
        delta.grad.zero_()
        
    facemask.data = torch.min(torch.max(facemask.detach(), -X*(1-mask)), 1-X*(1-mask))
    return facemask.detach()    
