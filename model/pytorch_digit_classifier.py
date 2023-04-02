import torch
import torch.nn as nn
import torch.nn.functional as F

IMG_SIZE = 32

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        kernel_size=5
        self.conv1=nn.Conv2d(in_channels=1, out_channels=32, kernel_size=kernel_size, stride=1, padding=0)
        self.conv2=nn.Conv2d(in_channels=32, out_channels=64, kernel_size=kernel_size, stride=1, padding=0)
        output_layer_size=((IMG_SIZE-kernel_size+1)-kernel_size+1)//2
        self.dropout1=nn.Dropout(0.25)
        self.dropout2=nn.Dropout(0.5)
        self.fc1=nn.Linear(64*output_layer_size*output_layer_size,256)
        self.fc2=nn.Linear(256,10)
    
    def forward(self,x):
        x=self.conv1(x)
        x=F.relu(x)
        x=self.conv2(x)
        x=F.relu(x)
        x=F.max_pool2d(x,2)
        x=self.dropout1(x)
        x=torch.flatten(x,1)
        x=self.fc1(x)
        x=F.relu(x)
        x=self.dropout2(x)
        x=self.fc2(x)
        return F.log_softmax(x, dim=1)


