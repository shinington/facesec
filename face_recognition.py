"""
face_recognition.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Face recognition systems to be evaluated.
Can be either open-set or closed-set face recognition.
Can be either source model or target model.
"""


import os 
import torch.optim as optim
import torch
import torch.nn as nn
import time
import copy
import numpy as np

from architecture.vggface import get_pretrained_vggface
from architecture.facenet import get_pretrained_facenet
from attack import utgt_pgd, utgt_occlusion, utgt_facemask
from data import get_dataloader, store_dataloader
from config import config
from utils import load_mask

class FaceRecognition():
    """
    Face recognition systems to be evaluated.
    """
    def __init__(self, mode, model_name, test_datapath=None, gallery_datapath=None, output_datapath=None, univ=False):
        """
        Initialize experiments.
        Args:
            mode (string): one of ['closed', 'open'].
            model_name (string): the name of model to be evaluated. 
            test_datapath (string): directory of the test set.
            gallery_datapath (string): directory of the gallery set (only for open-set face recognition, by default it is an empty string). 
            output_datapath (string): directory of the produced adversarial examples.
            univ (bool): whether to produce universal adversarial examples at testing time.
        """
        self.mode = mode
        self.test_datapath = test_datapath
        self.gallery_datapath = gallery_datapath 
        self.output_datapath = output_datapath
        self.model_name = model_name
        self.univ = univ


    def load_model(self, ):
        """
        Load face recognition model.
        Args:
            model_name (string): name of the face recognition model to be evaluated. Used to switch models. 
        """
        
        # Load neural architecture.
        if self.model_name == 'vggface':
            self.model = get_pretrained_vggface(self.mode)
        elif self.model_name == 'facenet':
            self.model = get_pretrained_facenet(self.mode)

        # Load weights
        # Weights for open-set face recognition is loaded automatically when loading the architecture.
        # The weights of closed-set face recognition should be loaded explicitly, 
        # as we add a fully connected layer for classification
        if self.mode == 'closed':
            model_path = os.path.join('model/', '{}_closed.pt'.format(self.model_name))
            self.model.load_state_dict(torch.load(model_path))

        # The following step can significantly reduce GPU usage when performing attacks.
        for param in self.model.parameters():
            param.requires_grad = False

    def eval_folder_closed(self, data_path=None):
        """
        Evaluate closed-set face recognition on data stored in test_datapath.
        Metric: prediction accuracy
        """ 
        self.model.eval()

        if data_path == None:
            data_path = self.test_datapath
        test_loader = get_dataloader(data_path, config['batch_size'])

        running_corrects = 0.0    

        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.cuda(), y.cuda()

                # Forward
                ys = self.model(X)
                _, yp = torch.max(ys, 1)
                
                # Statistics
                running_corrects += torch.sum(yp == y.data)
                
        acc = running_corrects.double() / len(test_loader.dataset)       
        print("Accuracy on data stored in {}: {}".format(data_path, acc))

    def eval_folder_open(self, data_path=None):
        """
        Evaluate open-set face recognition on data stored in test_datapath.
        Metric: prediction accuracy
        """ 
        self.model.eval()

        if data_path == None:
            data_path = self.test_datapath
        test_loader = get_dataloader(data_path, config['batch_size'])
        gallery_loader = get_dataloader(self.gallery_datapath, config['batch_size'])

        running_corrects = 0.0    
        cos = nn.CosineSimilarity(dim=1, eps=1e-8)
        threshold = config['threshold'][self.model_name]
        
        with torch.no_grad():
            for (X_test, _), (X_gallery, _) in zip(test_loader, gallery_loader):
                X_test, X_gallery = X_test.cuda(), X_gallery.cuda()
                
                embedding_test = self.model(X_test)
                embedding_gallery = self.model(X_gallery)
                
                # Cosine similarity
                similarities = cos(embedding_test, embedding_gallery)
                
                for similarity in similarities:
                    if similarity >= threshold:
                        running_corrects += 1        

        acc = running_corrects / len(test_loader.dataset)       
        print("Accuracy on data stored in {}: {}".format(data_path, acc))

    def eval_folder(self, data_path=None):
        """
        Evaluate face recognition on data stored in test_datapath. 
        A wrapper for both closed-set and open-set.
        """
        if self.mode == 'closed':
            self.eval_folder_closed(data_path)
        elif self.mode == 'open':
            self.eval_folder_open(data_path)

    def eval_robustness_closed(self, attack):
        """
        Evaluate robustness of closed-set face recognition on adversarial examples.
        We first use white-box attacks to tramsform images in test_loader to adversarial examples.
        We Then evaluate robustness of face recognition against these.
        Metric: prediction accuracy
        Args:
            attack (string): attack method to produce adversarial examples. Can be one of ['pgd', 'eyeglass', 'sticker', 'facemask'].
        """ 
        self.model.eval()

        test_loader = get_dataloader(self.test_datapath, config['batch_size'])
        labels = sorted(os.listdir(self.test_datapath))
        running_corrects = 0.0    

        for batch_idx, (X, y) in enumerate(test_loader):
            X, y = X.cuda(), y.cuda()

            # Get adversarial example by performing white-box attack on self.model
            if attack == 'pgd':
                delta = utgt_pgd(self.mode, self.model, X, y, univ=self.univ)
                
                ys = self.model(X+delta)
                _, yp = torch.max(ys, 1)
                running_corrects += torch.sum(yp == y.data)

                if batch_idx == 0:
                    X_adv = X+delta
                    y_adv = y
                else:
                    X_adv = torch.cat((X_adv, X+delta), 0)
                    y_adv = torch.cat((y_adv, y), 0)
            else:
                mask = load_mask(attack).cuda()
                if attack != 'facemask':
                    delta = utgt_occlusion(self.mode, self.model, X, y, mask, univ=self.univ)
                else:
                    delta = utgt_facemask(self.mode, self.model, X, y, mask, univ=self.univ)
                
                ys = self.model(X*(1-mask)+delta)
                _, yp = torch.max(ys, 1)
                running_corrects += torch.sum(yp == y.data)

                if batch_idx == 0:
                    X_adv = X*(1-mask)+delta
                    y_adv = y
                else:
                    X_adv = torch.cat((X_adv, X*(1-mask)+delta), 0)
                    y_adv = torch.cat((y_adv, y), 0)
            
        # Get robustness = prediction accuracy    
        acc = running_corrects.double() / len(test_loader.dataset)       
        print("Accuracy on adversarial examples to be stored in {}: {}".format(self.output_datapath, acc))

        # Store adversarial examples for future use.
        adv_dataset = torch.utils.data.TensorDataset(X_adv.cpu().detach(), y_adv.cpu().detach())
        adv_loader = torch.utils.data.DataLoader(adv_dataset, batch_size=1, shuffle=False)
        store_dataloader(adv_loader, self.output_datapath, labels)

    def eval_robustness_open(self, attack):
        """
        Evaluate robustness of open-set face recognition on adversarial examples.
        We first use white-box attacks to tramsform images in test_loader to adversarial examples.
        We Then evaluate robustness of face recognition against these.
        Metric: prediction accuracy
        Args:
            attack (string): attack method to produce adversarial examples. Can be one of ['pgd', 'eyeglass', 'sticker', 'facemask'].
        """ 
        self.model.eval()

        test_loader = get_dataloader(self.test_datapath, config['batch_size'])
        gallery_loader = get_dataloader(self.gallery_datapath, config['batch_size'])
        labels = sorted(os.listdir(self.test_datapath))

        running_corrects = 0.0    
        cos = nn.CosineSimilarity(dim=1, eps=1e-8)
        threshold = config['threshold'][self.model_name]

        for batch_idx, ((X_test, y_test), (X_gallery, _)) in enumerate(zip(test_loader, gallery_loader)):
            X_test, y_test, X_gallery = X_test.cuda(), y_test.cuda(), X_gallery.cuda()

            # Get adversarial example by performing white-box attack on self.model
            if attack == 'pgd':
                delta = utgt_pgd(self.mode, self.model, X_test, y_test, univ=self.univ)
                similarities = cos(self.model(X_test+delta), self.model(X_gallery))

                if batch_idx == 0:
                    X_adv = X_test+delta
                    y_adv = y_test
                else:
                    X_adv = torch.cat((X_adv, X_test+delta), 0)
                    y_adv = torch.cat((y_adv, y_test), 0)
            else:
                mask = load_mask(attack).cuda()
                if attack != 'facemask':
                    delta = utgt_occlusion(self.mode, self.model, X_test, y_test, mask, univ=self.univ)
                else:
                    delta = utgt_facemask(self.mode, self.model, X_test, y_test, mask, univ=self.univ)
                similarities = cos(self.model(X_test*(1-mask)+delta), self.model(X_gallery))

                if batch_idx == 0:
                    X_adv = X_test*(1-mask)+delta
                    y_adv = y_test
                else:
                    X_adv = torch.cat((X_adv, X_test*(1-mask)+delta), 0)
                    y_adv = torch.cat((y_adv, y_test), 0)
            
            for similarity in similarities:
                if similarity >= threshold:
                    running_corrects += 1        

        # Get robustness = prediction accuracy    
        acc = running_corrects / len(test_loader.dataset)       
        print("Accuracy on adversarial examples to be stored in {}: {}".format(self.output_datapath, acc))

        # Store adversarial examples for future use.
        adv_dataset = torch.utils.data.TensorDataset(X_adv.cpu().detach(), y_adv.cpu().detach())
        adv_loader = torch.utils.data.DataLoader(adv_dataset, batch_size=1, shuffle=False)
        store_dataloader(adv_loader, self.output_datapath, labels)

    def eval_robustness(self, attack):
        """
        Evaluate face recognition with adversairal examples. 
        A wrapper for both closed-set and open-set.
        """
        if self.mode == 'closed':
            self.eval_robustness_closed(attack)
        elif self.mode == 'open':
            self.eval_robustness_open(attack)
