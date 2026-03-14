import torch
import torch.nn as nn
import torch.nn.functional as F

class Simple3DConvNet1(nn.Module):
    def __init__(self, in_channels=3, mid_channels=16):
        super(Simple3DConvNet1, self).__init__()
        # convolutional window 5 frames x 11 x 11

        self.conv1 = nn.Conv3d(in_channels, 4*mid_channels, kernel_size=(5,11,11), padding=(2,5,5), stride=(1, 2, 2))
        self.bn1 = nn.BatchNorm3d(4*mid_channels)

        self.avgpool = nn.AvgPool3d(kernel_size=(1, 4, 4), stride=(1, 4, 4))
        
    def forward(self, x):
        x1 = F.relu(self.bn1(self.conv1(x)))  
        x1 = self.avgpool(x1)
        
        return x1
        
class Simple3DConvNet3(nn.Module):
    def __init__(self, in_channels=3, mid_channels=16):
        super(Simple3DConvNet3, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, 4*mid_channels, kernel_size=(5,11,11), padding=(2,5,5), stride=(1, 2, 2))
        self.bn1 = nn.BatchNorm3d(4*mid_channels)
        self.conv2 = nn.Conv3d(4*mid_channels, 4*mid_channels, kernel_size=3, padding=1, stride=(1, 2, 2))
        self.bn2 = nn.BatchNorm3d(4*mid_channels)
        self.conv3 = nn.Conv3d(4*mid_channels, 4*mid_channels, kernel_size=3, padding=1, stride=(1, 2, 2))
        self.bn3 = nn.BatchNorm3d(4*mid_channels)
        
    def forward(self, x):
        x1 = F.relu(self.bn1(self.conv1(x)))  
        x2 = F.relu(self.bn2(self.conv2(x1)))  
        x3 = F.relu(self.bn3(self.conv3(x2)))
        
        return x3
        
class Simple3DConvNet5(nn.Module):
    def __init__(self, in_channels=3, mid_channels=16):
        super(Simple3DConvNet5, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, 4*mid_channels, kernel_size=(5,11,11), padding=(2,5,5), stride=(1, 2, 2))
        self.bn1 = nn.BatchNorm3d(4*mid_channels)
        self.conv2 = nn.Conv3d(4*mid_channels, 4*mid_channels, kernel_size=3, padding=1, stride=(1, 2, 2))
        self.bn2 = nn.BatchNorm3d(4*mid_channels)
        self.conv3 = nn.Conv3d(4*mid_channels, 4*mid_channels, kernel_size=3, padding=1, stride=(1, 2, 2))
        self.bn3 = nn.BatchNorm3d(4*mid_channels)
        self.conv4 = nn.Conv3d(4*mid_channels, 4*mid_channels, kernel_size=3, padding=1, stride=(1, 1, 1))
        self.bn4 = nn.BatchNorm3d(4*mid_channels)
        self.conv5 = nn.Conv3d(4*mid_channels, 4*mid_channels, kernel_size=3, padding=1, stride=(1, 1, 1))
        self.bn5 = nn.BatchNorm3d(4*mid_channels)
        
    def forward(self, x):
        x1 = F.relu(self.bn1(self.conv1(x)))  
        x2 = F.relu(self.bn2(self.conv2(x1)))  
        x3 = F.relu(self.bn3(self.conv3(x2)))
        x4 = F.relu(self.bn4(self.conv4(x3)))  
        x5 = F.relu(self.bn5(self.conv5(x4)))
        return x5

class Simple3DConvNet5(nn.Module):
    def __init__(self, in_channels=3, mid_channels=16):
        super(Simple3DConvNet5, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, 4*mid_channels, kernel_size=(5,11,11), padding=(2,5,5), stride=(1, 2, 2))
        self.bn1 = nn.BatchNorm3d(4*mid_channels)
        self.proj1 = nn.Conv3d(in_channels, 4*mid_channels, kernel_size=1, stride=(1,2,2))
        
        self.conv2 = nn.Conv3d(4*mid_channels, 4*mid_channels, kernel_size=3, padding=1, stride=(1, 2, 2))
        self.bn2 = nn.BatchNorm3d(4*mid_channels)
        self.proj2 = nn.Conv3d(4*mid_channels, 4*mid_channels, kernel_size=1, stride=(1,2,2))
        
        self.conv3 = nn.Conv3d(4*mid_channels, 4*mid_channels, kernel_size=3, padding=1, stride=(1, 2, 2))
        self.bn3 = nn.BatchNorm3d(4*mid_channels)
        self.proj3 = nn.Conv3d(4*mid_channels, 4*mid_channels, kernel_size=1, stride=(1,2,2))
        
        self.conv4 = nn.Conv3d(4*mid_channels, 4*mid_channels, kernel_size=3, padding=1, stride=(1, 1, 1))
        self.bn4 = nn.BatchNorm3d(4*mid_channels)
        
        self.conv5 = nn.Conv3d(4*mid_channels, 4*mid_channels, kernel_size=3, padding=1, stride=(1, 1, 1))
        self.bn5 = nn.BatchNorm3d(4*mid_channels)
        
    def forward(self, x):
        x1 = F.gelu(self.bn1(self.conv1(x))) + self.proj1(x)
        x2 = F.gelu(self.bn2(self.conv2(x1))) + self.proj2(x1)
        x3 = F.gelu(self.bn3(self.conv3(x2))) + self.proj3(x2)
        x4 = F.gelu(self.bn4(self.conv4(x3))) + x3  
        x5 = F.gelu(self.bn5(self.conv5(x4))) + x4  
        return x5



class Simple3DConvNet7(nn.Module):
    def __init__(self, in_channels=3, mid_channels=16):
        super(Simple3DConvNet7, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, 4*mid_channels, kernel_size=(5,11,11), padding=(2,5,5), stride=(1, 2, 2))
        self.bn1 = nn.BatchNorm3d(4*mid_channels)
        self.conv2 = nn.Conv3d(4*mid_channels, 4*mid_channels, kernel_size=3, padding=1, stride=(1, 2, 2))
        self.bn2 = nn.BatchNorm3d(4*mid_channels)
        self.conv3 = nn.Conv3d(4*mid_channels, 4*mid_channels, kernel_size=3, padding=1, stride=(1, 2, 2))
        self.bn3 = nn.BatchNorm3d(4*mid_channels)
        self.conv4 = nn.Conv3d(4*mid_channels, 4*mid_channels, kernel_size=3, padding=1, stride=(1, 1, 1))
        self.bn4 = nn.BatchNorm3d(4*mid_channels)
        self.conv5 = nn.Conv3d(4*mid_channels, 4*mid_channels, kernel_size=3, padding=1, stride=(1, 1, 1))
        self.bn5 = nn.BatchNorm3d(4*mid_channels)
        self.conv6 = nn.Conv3d(4*mid_channels, 4*mid_channels, kernel_size=3, padding=1, stride=(1, 1, 1))
        self.bn6 = nn.BatchNorm3d(4*mid_channels)
        self.conv7 = nn.Conv3d(4*mid_channels, 4*mid_channels, kernel_size=3, padding=1, stride=(1, 1, 1))
        self.bn7 = nn.BatchNorm3d(4*mid_channels)
        
    def forward(self, x):
        x1 = F.relu(self.bn1(self.conv1(x)))  
        x2 = F.relu(self.bn2(self.conv2(x1)))  
        x3 = F.relu(self.bn3(self.conv3(x2)))  
        x4 = F.relu(self.bn4(self.conv4(x3)))  
        x5 = F.relu(self.bn5(self.conv5(x4)))
        x6 = F.relu(self.bn6(self.conv6(x5)))
        x7 = F.relu(self.bn7(self.conv7(x6)))
        
        return x7
        
class Simple3DResNet5(nn.Module):
    def __init__(self, in_channels=3, mid_channels=8):
        super(Simple3DResNet5, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=5, padding=2, stride=(1, 2, 2))
        self.bn1 = nn.BatchNorm3d(mid_channels)
        self.conv2 = nn.Conv3d(mid_channels, 2*mid_channels, kernel_size=3, padding=1, stride=(1, 2, 2))
        self.bn2 = nn.BatchNorm3d(2*mid_channels)
        
        self.conv3 = nn.Conv3d(2*mid_channels, 2*mid_channels, kernel_size=3, padding=1, stride=(1,1,1))
        self.bn3 = nn.BatchNorm3d(2*mid_channels)
        self.conv4 = nn.Conv3d(2*mid_channels, 2*mid_channels, kernel_size=3, padding=1, stride=(1,1,1))
        self.bn4 = nn.BatchNorm3d(2*mid_channels)
        
        self.conv5 = nn.Conv3d(2*mid_channels, 4*mid_channels, kernel_size=3, padding=1, stride=(1, 2, 2))
        self.bn5 = nn.BatchNorm3d(4*mid_channels)
        
    def forward(self, x):
        x1 = F.relu(self.bn1(self.conv1(x)))  
        x2 = F.relu(self.bn2(self.conv2(x1)))  
        
        x3 = F.relu(self.bn3(self.conv3(x2)))
        x4 = F.relu(x2 + self.bn4(self.conv4(x3)))
        
        x5 = F.relu(self.bn5(self.conv5(x4)))
        
        return x5