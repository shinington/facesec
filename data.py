import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from PIL import Image

def get_dataloader(data_path, batch_size):
    """
    Read images from a directory, convert these into torch.Tensor , and then store them in a dataloader.
    Args:
        data_path (string): the directory of images used in evaluation.
        batch_size (int): size of each mini batch used for evaluation.
    Returns:
        dataloader (torch DataLoader): the data loader for evaluation.
    """
    data_transforms =  transforms.Compose([
    transforms.ToTensor(),
    ])
    
    image_datasets = datasets.ImageFolder(data_path, data_transforms) 
    dataloader = torch.utils.data.DataLoader(image_datasets, batch_size=batch_size, shuffle=False)
    
    dataset_size = len(image_datasets) 
    #print("Size of dataset: ", dataset_size)
    return dataloader

def plot_sample(X, M, N):
    """
    Plot a subset of a dataset given the tensor of the feature vectors.
    Note that M*N should be less then X.shape[0].
    Args:
        X (torch.Tensor): the image batch to plot.
        M (int): rows of image to plot.
        N (int): columns of image to plot.
    """
    
    #%matplotlib inline

    print("Show the images...")
    f,ax = plt.subplots(M, N, sharex=True, sharey=True, figsize=(N*3, M*3))
    for i in range(M):
        for j in range(N):
            ax[i][j].imshow(X[i*N+j].cpu().detach().numpy().transpose((1, 2, 0)))
            ax[i][j].set_axis_off()
    plt.tight_layout()
    #plt.show();

def store_dataloader(data_loader, data_path, labels):
    """
    Store images returned by data_loader in data_path, categorized by labels.
    Args:
        data_loader (torch.utils.data.dataloader.DataLoader): image dataloader, batchsize should be 1.
        data_path (string): directory to store images.
        labels (list): list of identities (in ASCII order).
    """
    if not os.path.exists(data_path):
        os.system('mkdir {}'.format(data_path))
        # Create subfolders for each identity
        for label in labels:
            os.system('mkdir {}/{}'.format(data_path, label))

    # Then store images returned by data_loader.
    i = 0
    for X, y in data_loader:
        X = X[0].cpu().detach().numpy().transpose((1, 2, 0))
        X = np.clip(X, 0, 1)*255
        X = Image.fromarray(X.astype('uint8'))
        X.save('{}/{}/{}.png'.format(data_path, labels[int(y[0])], i))
        i += 1
