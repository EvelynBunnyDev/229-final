import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
        
class GaborPyramidModule(nn.Module):
    """
    Create a module that maps stacks of images to a 3d Gabor pyramid.
    Only works in grayscale.
    from https://github.com/patrickmineault/your-head-is-there-to-move-you-around/blob/main/modelzoo/gabor_pyramid.py
    """
    def __init__(self, 
                 nlevels=5,
                 nt=5,
                 stride=4, ct='complex'):
        super(GaborPyramidModule, self).__init__()
        self.nt = nt
        self.nlevels = nlevels
        self.stride = stride
        self.ct = ct
        self.setup()
        
    def setup(self):
        # The filters will be 1x1xntx9x9 - no is the number of orientations
        nx, no = 9, 4
        zi, yi, xi = torch.meshgrid(torch.arange(-(self.nt // 2), 
                                                 (self.nt + 1) // 2), 
                                    torch.arange(-4, 5), 
                                    torch.arange(-4, 5), indexing='ij')

        assert zi.shape[0] == self.nt
        filters = []
        for ii in range(no):
            # static + two directions 
            for dt in [-1, 0, 1]:
                coso = np.cos(ii * np.pi / no)
                sino = np.sin(ii * np.pi / no)
                G = torch.exp(-(xi**2 + yi**2 + (zi / self.nt * nx)**2)/2/2**2)
                thefilt1 = torch.cos((coso * xi + sino * yi)*.8 + dt * zi / self.nt * np.pi * 2) * G
                thefilt2 = torch.sin((coso * xi + sino * yi)*.8 + dt * zi / self.nt * np.pi * 2) * G
                thefilt1 = thefilt1 - G / G.mean() * thefilt1.mean()
                thefilt2 = thefilt2 - G / G.mean() * thefilt2.mean()
                scale = 1 / torch.sqrt((thefilt1 ** 2).sum())

                filters += [thefilt1 * scale, thefilt2 * scale]

        downsample_filt = torch.tensor([[.25, .5, .25], [.5, 1.0, .5], [.25, .5, .25]]).view(1, 1, 3, 3)
        downsample_filt /= 4.0

        filters = torch.stack(filters, dim=0).view(no * 3 * 2, 1, self.nt, nx, nx)
        self.register_buffer('filters', filters, False)
        self.register_buffer('downsample_filt', downsample_filt, False)

    def forward(self, X):
        # Transform to grayscale.
        X_ = X.sum(axis=1, keepdims=True)
        maps = []
        for i in range(self.nlevels):
            stride = int(self.stride/(2 ** i) if self.stride/(2 ** i) > 1 else 1)
            
            outputs = F.conv3d(X_, 
                               self.filters, 
                               padding=(self.nt//2, 4, 4), # padding is 2, 4, 4
                               stride=(1, stride, stride))
            if self.ct == 'complex':
                magnitude = torch.sqrt((outputs ** 2)[:, ::2, :, :, :] + 
                                   (outputs ** 2)[:, 1::2, :, :, :])
            elif self.ct == 'simple':
                magnitude = F.relu(outputs)
            else:
                raise ValueError("invalid cell type")

            if i <= 2:
                maps.append(magnitude)
            else:
                # Only the spatial dimension is resized.
                the_map = F.interpolate(magnitude.reshape((magnitude.shape[0], -1, magnitude.shape[-2], magnitude.shape[-1])), 
                                        scale_factor=2**(i-2), 
                                        mode='bilinear', 
                                        align_corners=False)
                the_map = the_map.reshape(magnitude.shape[0], 
                                          magnitude.shape[1], -1, the_map.shape[-2], the_map.shape[-1])[:, :, :, :maps[0].shape[-2], :maps[0].shape[-1]]
                maps.append(the_map)

            X_ = F.conv2d(X_.reshape((X_.shape[0]*X_.shape[2], 1, X_.shape[-2], X_.shape[-1])), 
                    self.downsample_filt, 
                    padding=1, 
                    stride=2)
            X_ = X_.reshape(X.shape[0], 1, -1, X_.shape[-2], X_.shape[-1])
            
        return torch.cat(maps, axis=1)

class GaborPyramid3d(nn.Module):
    """
    Create a module that maps stacks of images to a 3d Gabor pyramid.
    Only works in grayscale
    from https://github.com/patrickmineault/your-head-is-there-to-move-you-around/blob/main/modelzoo/gabor_pyramid.py
    """
    def __init__(self, 
                 nlevels=5,
                 nt=5,
                 stride=4):
        super(GaborPyramid3d, self).__init__()
        self.layer1 = GaborPyramidModule(nlevels=nlevels, nt=nt, stride=stride, ct='simple')
        self.layer2 = GaborPyramidModule(nlevels=nlevels, nt=nt, stride=stride, ct='complex')

    def forward(self, X):
        simp_out = self.layer1(X)
        comp_out = self.layer2(X)
        return comp_out