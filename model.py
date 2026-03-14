import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Dict, Iterable, Callable
import numpy as np
import torch
from torch import nn, Tensor, FloatTensor
import torch.nn.functional as F

from typing import Dict, Iterable, Callable

from PIL import Image
import numpy as np
from math import exp
from scipy import optimize

from utils import set_seed
from baselines.gaborpyramid.gabor_pyramid import GaborPyramid3d
from baselines.dorsalnet.dorsal_net import DorsalNet, extract_subnet_dict
from baselines.simple3d import Simple3DConvNet1, Simple3DConvNet3, Simple3DConvNet5, Simple3DResNet5, Simple3DConvNet7
from torchvision.models import resnet18, ResNet18_Weights, googlenet, GoogLeNet_Weights
from torchvision.models.video import r3d_18, R3D_18_Weights

class VideoModel(nn.Module):
    """
    Model of neural responses response to videos
    # based on https://www.cell.com/neuron/fulltext/S0896-6273(24)00881-X
    """

    def __init__(self, pretrained_model, layer, n_neurons, device=None, input_shape=(1, 3, 5, 224, 224), endtoend=False):
        super(VideoModel, self).__init__()

        self.layer = layer

        self.endtoend = endtoend
        
        # fix parameters of pretrained model
        if not self.endtoend:
            for param in pretrained_model.parameters():
                param.requires_grad = False

        self.ann = pretrained_model

        # get feature extractor 
        if not self.endtoend:
            self.inc_features = FeatureExtractor(self.ann, layers=[self.layer])
        else:
            self.inc_features = pretrained_model
        self.inc_features.to(device)

        dummy_input = torch.ones(size=input_shape) # we need to figure out the shape of the output layer
        if device is not None:
            dummy_input = dummy_input.to(device)

        dummy_feats = self.inc_features(dummy_input)
        if not self.endtoend:
            self.mod_shape = dummy_feats[self.layer].shape
        else:
            self.mod_shape = dummy_feats.shape

        if len(self.mod_shape) == 4: # if does not have batch dimension, add it
            self.mod_shape = dummy_feats[self.layer].unsqueeze(0).shape

        print("input shape to readout: " + self.mod_shape.__str__())

        # create w_s and w_f
        self.w_s = torch.nn.Parameter(torch.ones(n_neurons, 1, self.mod_shape[-1]*self.mod_shape[-2], 1, 
                                                  device=device,requires_grad=True))
        self.w_f = torch.nn.Parameter(torch.ones(1, n_neurons, 1, self.mod_shape[1]*self.mod_shape[2], 
                                                  device=device, requires_grad=True))

        # initialize w_s and w_f 
        nn.init.xavier_normal_(self.w_f) # feature weights are xavier normal
        nn.init.xavier_normal_(self.w_s) # spatial weights are xavier normal 
        
        # create a fixed batch norm layer 
        self.ann_bn = torch.nn.BatchNorm2d(self.mod_shape[1]*self.mod_shape[2], affine=False) # num_features is num_chan*num_time

    def forward(self,x):
        x = self.inc_features(x) # extract features
        if not self.endtoend:
            x = x[self.layer] # get features at specific layer 
    
        x = x.view(x.shape[0], x.shape[1]*x.shape[2], x.shape[3], x.shape[4]) # put time and features together B X (TXC) X H X W
        x = self.ann_bn(x) # batch norm then relu
        x = x.view(x.shape[0],x.shape[1],x.shape[2]*x.shape[3],1) # reshape so that the spatial dimensions are in one dimension, B X (TXC) X (HXW) X 1
        x = x.permute(0,-1,2,1) # reorder for the elementwise product with the spatial weights, B X 1 X (HXW) X (TXC)
        x = F.conv2d(x,torch.abs(self.w_s)) # convolve with the abs val of spatial weights, w_s is N x 1 x (HXW) X 1, so the convolution yields B X N x 1 x (TXC)
        x = torch.mul(x,self.w_f) # multiply with feature weights elemtwise operation
        x = torch.sum(x,-1,keepdim=True) # sum over weighted features, last dim
        return x # B x N X 1

class ImageModel(nn.Module):
    """
    Model of neural responses response to images
    # based on https://www.cell.com/neuron/fulltext/S0896-6273(24)00881-X
    """

    def __init__(self, pretrained_model, layer, n_neurons, device=None, input_shape=(1, 3, 224, 224)):
        super(ImageModel, self).__init__()

        self.layer = layer

        # fix parameters of pretrained model
        for param in pretrained_model.parameters():
            param.requires_grad = False
        self.ann = pretrained_model

        # get feature extractor 
        self.inc_features = FeatureExtractor(self.ann, layers=[self.layer])
        dummy_input = torch.ones(size=input_shape) # we need to figure out the shape of the output layer

        if device is not None:
            self.inc_features.to(device)
            dummy_input = dummy_input.to(device)
            
        dummy_feats = self.inc_features(dummy_input)
        self.mod_shape = dummy_feats[self.layer].shape

        if len(self.mod_shape) == 3: # if does not have batch dimension, add it
            self.mod_shape = dummy_feats[self.layer].unsqueeze(0).shape
                    
        print("input shape to readout: " + self.mod_shape.__str__())
        
        # initialize our two parameters 
        # note this assumes square input/feature map
        self.w_s = torch.nn.Parameter(torch.ones(n_neurons, 1, self.mod_shape[-1]*self.mod_shape[-1], 1, 
                                                  device=device,requires_grad=True))
        self.w_f = torch.nn.Parameter(torch.ones(1, n_neurons, 1, self.mod_shape[1], 
                                                  device=device, requires_grad=True))

        # initialize w_s and w_f 
        nn.init.xavier_normal_(self.w_f) # feature weights are xavier normal
        nn.init.xavier_normal_(self.w_s) # spatial weights are xavier normal 
                              
        # initialize a batch norm layer
        self.ann_bn = torch.nn.BatchNorm2d(self.mod_shape[1], affine=False)

    def forward(self,x):
        x = self.inc_features(x) # extract features
        x = x[self.layer] # get features at specific layer 
        x = self.ann_bn(x) # batch norm then relu
        x = x.view(x.shape[0],x.shape[1],x.shape[2]*x.shape[3],1) # reshape so that the spatial dimensions are in one dimension
        x = x.permute(0,-1,2,1) # reorder for the convlution with the spatial weights
        x = F.conv2d(x,torch.abs(self.w_s)) # convolve with the spatial weights, negative values not allowed
        x = torch.mul(x,self.w_f) # multiply with feature weights
        x = torch.sum(x,-1,keepdim=True) # sum
        return x

class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}
        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def forward(self, x: Tensor):
        _ = self.model(x)
        return self._features

def get_pretrained_model(name):
    if name == 'resnet18':
        return resnet18(weights=ResNet18_Weights.DEFAULT)
    elif name == 'r3d18':
        return r3d_18(weights=R3D_18_Weights.DEFAULT)
    elif name == 'gabor3d':
        return GaborPyramid3d()
    elif name == 'simple3d1':
        return Simple3DConvNet1()
    elif name == 'simple3d3':
        return Simple3DConvNet3()
    elif name == 'simple3d5':
        return Simple3DConvNet5()
    elif name == 'simple3d7':
        return Simple3DConvNet7()
    elif name == 'simple3d5res':
        return Simple3DResNet5()
    elif name == 'dorsalnet':
        net = DorsalNet(False, 32)
        dorsalnet_ckpt_path = './baselines/dorsalnet/checkpoints/dorsalnet.pt'
        subnet_dict = extract_subnet_dict(torch.load(dorsalnet_ckpt_path, map_location=torch.device('cpu')))
        net.load_state_dict((subnet_dict))
        return net
    else:
        raise ValueError(f'Unknown model {name}')