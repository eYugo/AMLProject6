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
        print(f"x shape inside forward {x.shape}")
        for i in range(len(M)):
            print(f"M shape inside forward {i} {M[i].shape}")
        i = 0
        for layer in [self.resnet.conv1, self.resnet.bn1, self.resnet.relu, self.resnet.maxpool] + layers_to_apply_shaping:
            x = layer(x)
            
            # task 1: apply activation shaping 
            #   to use uncomment the "if statement"
            #   default applies activation shaping after each layer in layers_to_apply_shaping, you can change this by changing the if statement
            
            # task 2: apply random activation shaping with a probability of 0.5
            #   to use uncomment the "if statement"
            #   just pass one argument to the activation_shaping function (M is None by default)
            
            # Apply activation shaping after each layer in layers_to_apply_shaping
            if M is None:
                if layer in layers_to_apply_shaping:
                    x = self.activation_shaping(x)
            else:
                if layer in layers_to_apply_shaping:
                    x = self.activation_shaping(x, M[i])
                    i += 1

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)
        
        return x

class ActivationShapingModule(nn.Module):
    def __init__(self, probability=0.5):
        super(ActivationShapingModule, self).__init__()
        self.probability = probability
        
    
    def forward(self, A, M=None):
        A_binary = torch.where(A > 0, torch.tensor(1.0, device=A.device), torch.tensor(0.0, device=A.device))
        if M is None:
            # torch.bernoulli returns a tensor with the same shape as A with elements that are 0 or 1
            M = torch.bernoulli(torch.full_like(A, self.probability, device=A.device))
        M_binary = torch.where(M > 0, torch.tensor(1.0, device=A.device), torch.tensor(0.0, device=A.device))
        
        print(f"A shape inside asm {A.shape}")
        print(f"M shape inside asm {M.shape}")
        
        return A_binary * M_binary

class ASHResNet18(BaseResNet18):
    def __init__(self):
        super(ASHResNet18, self).__init__()

    def forward(self, x, Mt=None):
        return super().forward(x, Mt)

    def get_activation_maps(self, xt):
        # Forward pass through the network to obtain activation maps for the target domain
        self.eval()
        with torch.no_grad():
            Mt = []
            for layer in [self.resnet.conv1, self.resnet.bn1, self.resnet.relu, self.resnet.maxpool,
                          self.resnet.layer1, self.resnet.layer2, self.resnet.layer3, self.resnet.layer4]:
                xt = layer(xt)
                Mt.append(xt.clone())
            return Mt
