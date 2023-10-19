import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18, resnet50
from Datasets import * 
from torch.utils.data import DataLoader
from tqdm import tqdm   
import time 
import sys 

class FCN(nn.Module):
    def __init__(self, **kwargs):

        super(FCN, self).__init__()

        ###### Input Hparams ########################
        self.input_dim          =  kwargs['input_dim']
        self.feature_dim        =  kwargs['feature_dim']
        self.network_dimensions =  kwargs['network_dimensions']
        self.loss_type          =  kwargs['loss_type']
        self.num_classes        =  kwargs['num_classes']

        self.network_dimensions = [self.input_dim] + [int(x) for x in self.network_dimensions.split("-")] + [self.feature_dim]
        ################################################
        
        self.f = []
        for i in range(len(self.network_dimensions) -1):
            fcl = nn.Linear(self.network_dimensions[i], self.network_dimensions[i+1], bias=True)
            self.f.append(fcl)
            if not (self.loss_type == "contrastive" and i == len(self.network_dimensions) -2): 
                self.f.append(nn.ReLU(inplace = True))                                       
        
        self.f.append(nn.Linear(self.network_dimensions[-1], self.num_classes)) # For regression set num_classes = 1 (or number of preds in case of multiple regr)
        self.f = nn.Sequential(*self.f)

    def forward(self, x, penult = -1, penult_all = False):
        if penult == -1 and not penult_all:
            logits = self.f(x)         
            return logits 
        elif penult_all == False:
            output = self.f[:-2](x)[:,penult]
            return output 
        elif penult_all == True:
            output = self.f[:-2](x)
            return output 
        
    def replace_relu_softplus(self, beta):
        for i in range(len(self.f)):
            module = self.f[i]
            if isinstance(module, nn.ReLU):
                self.f[i] = SoftPlus(beta)

    def replace_softplus_relu(self):
        for i in range(len(self.f)):
            module = self.f[i]
            if isinstance(module, SoftPlus):  
                self.f[i] = torch.nn.ReLU() 




##### ResNet Class
##### kwargs: (1) pretrained - BOOL load pretrained model (2) resnet - STR which resnet e.g. resnet18, resnet50, etc (3) cifarstyle - BOOL do we want cifarstyle resnet i.e. remove first maxpool etc (4) num_classes INT dim of network outputs 
class ResNet(nn.Module):
    def __init__(self, **kwargs):
        super(ResNet, self).__init__() 

        # arguments passed in to resnet 
        self.pretrained      = kwargs['pretrained'] # bool if we want pretrained or not 
        self.resnet          = kwargs['resnet']     # str which resnet to use 
        self.cifarstyle      = kwargs['cifarstyle'] # bool - cifarstyle resnet w/ smaller conv + no pool
        self.num_classes     = kwargs['num_classes']

        self.f               = [] 

        self.selection_layer = None 
        self.softmax         = None 
        self.penult_forward = None 

        for name, module in getattr(sys.modules[__name__], self.resnet)(pretrained = self.pretrained, num_classes = self.num_classes).named_children():
            if self.cifarstyle:
                if name == 'conv1':
                    module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                    self.f.append(module)
                elif not isinstance(module, nn.MaxPool2d) and not isinstance(module, nn.Linear):
                    self.f.append(module)
                elif isinstance(module, nn.Linear):
                    self.linear = module 
            else:
                if  not isinstance(module, nn.Linear):
                    self.f.append(module)
                elif isinstance(module, nn.Linear):
                    self.linear = module 

        self.f = nn.Sequential(*self.f)

    def forward(self, x, penult = -1):
        feature = self.f(x)
        feature = feature.flatten(start_dim = 1)

        if penult != -1:
            return feature[:,penult]

        logits = self.linear(feature)
        if not self.softmax is None:
            logits = self.softmax(logits)
        if not self.selection_layer is None:
            logits = self.selection_layer(logits)
            logits = logits.squeeze(dim = 1) 
        return logits 

    def penult_forward_414(self, x):
        feature = self.f(x)
        feature = feature.flatten(start_dim = 1)
        return feature[:,414]

    # Replace each ReLU with SoftPlus parameterized by beta 
    def replace_relu_softplus(self, beta = 1):
        for name, module in self.named_modules():
            if hasattr(module, "relu"):
                module.relu = SoftPlus(beta = beta)
        self.f[2] = SoftPlus(beta = beta)

    # Replace each SoftPlus with ReLU 
    def replace_softplus_relu(self):
        for name, module in self.named_modules():
            if hasattr(module, "relu"):
                module.relu = torch.nn.ReLU(inplace = True)
        self.f[2] = torch.nn.ReLU(inplace = True)

###### Adapted from code for Integrated Hessian: 
###### https://github.com/suinleelab/path_explain/blob/b567945fe02ab8a6e6de675c8367edd45287b243/path_explain/utils.py Line 47 
class SoftPlus(torch.nn.Module):
    def __init__(self, beta):
        super().__init__()
        self.beta = beta 
    def forward(self, x):
        return (1.0 / self.beta) * torch.log(1.0 + \
                torch.exp(-1.0 * torch.abs(self.beta * x))) + \
                torch.maximum(x, torch.tensor(0).cuda() )