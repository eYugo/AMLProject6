import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class BaseResNet18(nn.Module):
    def __init__(self):
        super(BaseResNet18, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)
        self.activation_shaping = ActivationShapingModule()

    def forward(self, x, M=None):
        # Define the layers where activation shaping should be applied
        layers_to_apply_shaping = [self.resnet.layer1, self.resnet.layer2, self.resnet.layer3, self.resnet.layer4]
        for layer in [self.resnet.conv1, self.resnet.bn1, self.resnet.relu, self.resnet.maxpool] + layers_to_apply_shaping:
            x = layer(x)
            
            # task 1: apply activation shaping 
            #   to use uncomment the "if statement"
            #   default applies activation shaping after each layer in layers_to_apply_shaping, you can change this by changing the if statement
            
            # task 2: apply random activation shaping with a probability of 0.5
            #   to use uncomment the "if statement"
            #   just pass one argument to the activation_shaping function (M is None by default)
            
            # Apply activation shaping after each layer in layers_to_apply_shaping
            
            if layer in layers_to_apply_shaping:
                x = self.activation_shaping(x, M)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)
        
        return x

class ActivationShapingModule(nn.Module):
    def __init__(self, probability=0.0):
        super(ActivationShapingModule, self).__init__()
        self.probability = probability
        
    
    def forward(self, A, M=None):
        A_binary = torch.where(A > 0, torch.tensor(1.0, device=A.device), torch.tensor(0.0, device=A.device))
        if M is None:
            # torch.bernoulli returns a tensor with the same shape as A with elements that are 0 or 1
            M = torch.bernoulli(torch.full_like(A, self.probability, device=A.device))
        M_binary = torch.where(M > 0, torch.tensor(1.0, device=A.device), torch.tensor(0.0, device=A.device))
        
        if A_binary.shape != M_binary.shape:
            raise RuntimeError(f"Dimension mismatch: A_binary shape {A_binary.shape} must match M_binary shape {M_binary.shape}")
        
        return A_binary * M_binary

# class ASHResNet18(nn.Module):
#     def __init__(self):
#         super(ASHResNet18, self).__init__()
#         self.resnet = resnet18(weights=ResNet18_Weights)
#         self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)
#         self.activation_maps = {}

#     def forward(self, x):
#         layers_to_apply_shaping = [self.resnet.layer1, self.resnet.layer2, self.resnet.layer3, self.resnet.layer4]
#         for layer in [self.resnet.conv1, self.resnet.bn1, self.resnet.relu, self.resnet.maxpool] + layers_to_apply_shaping:
#             x = layer(x)

#             if layer == self.resnet.layer1:
                
#                 Mt = torch.where(x > 0, torch.tensor(1.0, device=x.device), torch.tensor(0.0, device=x.device))
#                 self.activation_maps[layer] = Mt.clone().detach()
                
#                 # if targ is True then register the activation map, otherwise apply the activation shaping
#                 x_bin = torch.where(x > 0, torch.tensor(1.0, device=x.device), torch.tensor(0.0, device=x.device))
#                 #m_bin = self.activation_maps[layer]
#                 x = x_bin * Mt

#         x = self.resnet.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.resnet.fc(x)

#         return x

class ASHResNet18(nn.Module):
    def __init__(self):
        super(ASHResNet18, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)
        self.activation_maps = {}

        self.record_layers = {
            'conv1': self.resnet.conv1,
            'bn1': self.resnet.bn1,
            'relu': self.resnet.relu,
            'maxpool': self.resnet.maxpool,
            'layer1': self.resnet.layer1,
            'layer2': self.resnet.layer2,
            'layer3': self.resnet.layer3,
            'layer4': self.resnet.layer4
        }

        self.activation_maps = {layer_name: None for layer_name in self.record_layers}

    def record_activation_maps(self, x):
        # Forward pass through each layer and store activations
        for layer_name, layer in self.record_layers.items():
            x = layer(x)
            self.activation_maps[layer_name] = x.clone().detach()

    def forward(self, x, target = None, test=True):
        
        if target is not None:
            self.record_activation_maps(target)
        
        for layer_name, layer in self.record_layers.items():
            x = layer(x)
            if test == False and layer_name == 'maxpool':
                mt = self.activation_maps[layer_name]
                mt_bin = torch.where(mt > 0, torch.tensor(1.0, device=mt.device), torch.tensor(0.0, device=mt.device))
                x_bin = torch.where(x > 0, torch.tensor(1.0, device=x.device), torch.tensor(0.0, device=x.device))
                x = x_bin * mt_bin

            
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)
        
        return x
    