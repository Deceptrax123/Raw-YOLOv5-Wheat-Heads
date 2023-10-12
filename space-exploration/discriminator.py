import torch 
from torch import nn 

from torch.nn import Module 
from torch.nn import Conv2d,BatchNorm2d,Flatten,Sigmoid,LeakyReLU,ConvTranspose2d,Linear,Dropout2d
from prettytable import PrettyTable

from torchsummary import summary

class Discriminator(Module):
    def __init__(self):
        super(Discriminator,self).__init__()

        self.conv1=Conv2d(in_channels=3,out_channels=8,stride=2,kernel_size=(3,3),padding=1)
        self.bn1=BatchNorm2d(8)
        self.dp1=Dropout2d(p=0.2)
        self.lr1=LeakyReLU(negative_slope=0.2)

        self.conv2=Conv2d(in_channels=8,out_channels=16,stride=2,kernel_size=(3,3),padding=1)
        self.dp2=Dropout2d(p=0.2)
        self.bn2=BatchNorm2d(num_features=16)
        self.lr2=LeakyReLU(negative_slope=0.2)

        self.conv3=Conv2d(in_channels=16,out_channels=32,kernel_size=(3,3),stride=2,padding=1)
        self.dp3=Dropout2d(p=0.2)
        self.bn3=BatchNorm2d(32)
        self.lr3=LeakyReLU(negative_slope=0.2)

        self.conv4=Conv2d(in_channels=32,out_channels=64,kernel_size=(3,3),stride=2,padding=1)
        self.dp4=Dropout2d(p=0.2)
        self.bn4=BatchNorm2d(64)
        self.lr4=LeakyReLU(negative_slope=0.2)

        self.conv5=Conv2d(in_channels=64,out_channels=128,kernel_size=(3,3),stride=2,padding=1)
        self.dp5=Dropout2d(p=0.2)
        self.bn5=BatchNorm2d(128)
        self.lr5=LeakyReLU(negative_slope=0.2)

        self.conv6=Conv2d(in_channels=128,out_channels=256,kernel_size=(3,3),stride=2,padding=1)
        self.dp6=Dropout2d(p=0.2)
        self.bn6=BatchNorm2d(256)
        self.lr6=LeakyReLU(negative_slope=0.2)

        self.conv7=Conv2d(in_channels=256,out_channels=512,kernel_size=(3,3),stride=2,padding=1)
        self.dp7=Dropout2d(p=0.2)
        self.bn7=BatchNorm2d(512)
        self.lr7=LeakyReLU(negative_slope=0.2)

        self.conv8=Conv2d(in_channels=512,out_channels=1024,kernel_size=(3,3),stride=2,padding=1)
        self.dp8=Dropout2d(p=0.2)
        self.bn8=BatchNorm2d(1024)
        self.lr8=LeakyReLU(negative_slope=0.2)

        self.conv9=Conv2d(in_channels=1024,out_channels=1,kernel_size=(4,4),padding=0,stride=1)

    
    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.dp1(x)
        x=self.lr1(x)

        x=self.conv2(x)
        x=self.dp2(x)
        x=self.bn2(x)
        x=self.lr2(x)

        x=self.conv3(x)
        x=self.dp3(x)
        x=self.bn3(x)
        x=self.lr3(x)

        x=self.conv4(x)
        x=self.dp4(x)
        x=self.bn4(x)
        x=self.lr4(x)

        x=self.conv5(x)
        x=self.dp5(x)
        x=self.bn5(x)
        x=self.lr5(x)

        x=self.conv6(x)
        x=self.dp6(x)
        x=self.bn6(x)
        x=self.lr6(x)

        x=self.conv7(x)
        x=self.dp7(x)
        x=self.bn7(x)
        x=self.lr7(x)

        x=self.conv8(x)
        x=self.dp8(x)
        x=self.bn8(x)
        x=self.lr8(x)

        x=self.conv9(x)

        return x

# model=Discriminator()
# summary(model,input_size=(3,1024,1024),batch_size=8,device='cpu')