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
# Hint: randomly sample 'target_examples' to obtain targ_x
#class DomainAdaptationDataset(Dataset):
#    def __init__(self, source_examples, target_examples, transform):
#        self.source_examples = source_examples
#        self.target_examples = target_examples
#        self.T = transform
#    
#    def __len__(self):
#        return len(self.source_examples)
#    
#    def __getitem__(self, index):
#        src_x, src_y = ...
#        targ_x = ...
#
#        src_x = self.T(src_x)
#        targ_x = self.T(targ_x)
#        return src_x, src_y, targ_x
######################################################