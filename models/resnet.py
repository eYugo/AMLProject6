import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class BaseResNet18(nn.Module):
    def __init__(self):
        super(BaseResNet18, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)

    def forward(self, x):
        return self.resnet(x)

class ActivationShapingModule(nn.Module):
    def __init__(self):
        super(ActivationShapingModule, self).__init__()

    def forward(self, A):
        # Generate random activation map M with the same shape as A with values in [0, 1]
        M = (torch.rand(A.shape)).to(A.device)

        # Binarize activation map A
        A_bin = torch.where(A > 0, torch.tensor(1.0), torch.tensor(0.0))
        # Binarize activation map M
        M_bin = torch.where(M >= 0, torch.tensor(1.0), torch.tensor(0.0))

        # Element-wise product of A and M
        shaped_output = A_bin * M_bin

        return shaped_output

class ASHResNet18(nn.Module):
    def __init__(self):
        super(ASHResNet18, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights)

        # Initialize ActivationShapingModule
        self.activation_shaping_module = ActivationShapingModule()
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        x = self.activation_shaping_module(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)

        return x

######################################################
# TODO: either define the Activation Shaping Module as a nn.Module
#class ActivationShapingModule(nn.Module):
#...
#
# OR as a function that shall be hooked via 'register_forward_hook'
#def activation_shaping_hook(module, input, output):
#...
#
######################################################
# TODO: modify 'BaseResNet18' including the Activation Shaping Module
#class ASHResNet18(nn.Module):
#    def __init__(self):
#        super(ASHResNet18, self).__init__()
#        ...
#    
#    def forward(self, x):
#        ...
#
######################################################