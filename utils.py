import torch
import numpy as np

from PIL import Image

def l1_norm(Z):
    """
    Compute the l1 norm of a given tensor. This is used to compute the momentum when solving optimization problems.
    Args:
        Z (torch.Tensor): input image with size [batch, 3, 224, 224].
    Returns:
        l1 norm of Z (torch.Tensor): reshaped l1 norm with size [batch, 1, 1, 1].    
    """
    return torch.sum(torch.abs(Z.view(Z.shape[0], -1)), 1)[:, None, None, None]

def load_mask(position):
    """
    Load the mask corresponding to the attack area. 
    Args:
        position (string): one of ['eyeglass', 'face', 'sticker'].
    Returns:
        mask (torch.Tensor): the mask. Size: 3*224*224.
    """
    mask = np.array(Image.open("mask/{}.png".format(position)))/255. # Read the image, the output format is channel last 
    mask = np.moveaxis(mask, 2, 0) # Channel last to channel first (default format in pytorch)
    mask = torch.from_numpy(mask).float() # From numpy to torch.tensor
    return mask

def facemask_matrix():
    """
    Return matrices that helps to initialize face mask attacks. 
    Returns:
        mask_left (torch.Tensor): mask for the left part of the 2-D facemask.
        mask_right (torch.Tensor): mask for the right part of the 2-D facemask.
        T_left (torch.Tensor): transform matrix for the left half part of face mask to 3-D.
        T_right (torch.Tensor): transform matrix for the right half part of face mask to 3-D. 
    """

    mask_left = torch.zeros(1, 3, 80, 160)
    mask_right = torch.zeros(1, 3, 80, 160)

    mask_left[:, :, :, 0:80] = 1
    mask_right[:, :, :, 80:160] = 1
    mask_left, mask_right = mask_left.cuda(), mask_right.cuda()


    T_left = torch.tensor([[[ 5.0323e-01,  0.0000e+00,  3.2000e+01],
             [-5.9355e-01,  1.0000e+00,  1.1200e+02],
             [-4.4355e-03, -0.0000e+00,  1.0000e+00]]]).cuda()
    T_right = torch.tensor([[[ 5.1556e+00,  0.0000e+00, -1.6356e+02],
             [ 2.0444e+00,  3.4444e+00,  5.8667e+01],
             [ 1.5278e-02,  0.0000e+00,  1.0000e+00]]]).cuda()
    return mask_left, mask_right, T_left, T_right

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
