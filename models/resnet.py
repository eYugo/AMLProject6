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



class ASHResNet18(nn.Module):
    def __init__(self):
        super(ASHResNet18, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)
        
        self.cas = CASLayer()
        self.handles = []

    def attach_forward_hook(self, Mt=None):
        count = 0
        for name, module in self.resnet.named_modules():
            if isinstance(module, nn.Conv2d) and count < 1:
                handle = module.register_forward_hook(
                    lambda module, input, output: self.cas.forward_hook(module, input, output, Mt))
                self.handles.append(handle)
                count += 1
    
    def detach_forward_hook(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def forward(self, x, test=True):
            if not test:
                self.attach_forward_hook()
                
            x = self.resnet(x)
            
            if not test:
                self.detach_forward_hook()
            return x

class CASLayer(nn.Module):
    def __init__(self):
        super(CASLayer, self).__init__()
        
    def forward_hook(self, module, input, output, Mt=None):
        print('Forward hook inside CASLayer is called')
        A = output.clone().detach()
        M = Mt.clone().detach() # if Mt is not None else torch.bernoulli(torch.full_like(A, 0.5, device=A.device))
        
        A_bin = torch.where(A > 0, torch.tensor(1.0, device=A.device), torch.tensor(0.0, device=A.device))
        M_bin = M.clone().detach() 
        
        return A_bin * M_bin

class RecordASLayer(nn.Module):
    def __init__(self):
        super(RecordASLayer, self).__init__()
    
    def forward_hook(self, module, input, output):
        # print('Recording activation')
        # print(f"name: {module.__class__.__name__}")
        A = output.clone().detach()
        A_bin = torch.where(A > 0, torch.tensor(1.0, device=A.device), torch.tensor(0.0, device=A.device))
        return A_bin

class ASHResNet18_rec(nn.Module):
    def __init__(self, layer_list=[]):
        super(ASHResNet18_rec, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)
        
        self.cas = CASLayer()
        self.rec = RecordASLayer()
        self.activation_maps = {}
        self.handles = []
        
        self.layer_list = layer_list
        for elem in self.layer_list:
            print(f"Executing ASM on layer: {elem}")

    def attach_forward_hook(self, Mt=None, target=False, frequency=2):
        counter = 0
        def create_hook_function(name, target, Mt):
            if target:
                return lambda module, input, output: self.record_activation(module, output, name)
            else:
                return lambda module, input, output: self.cas.forward_hook(module, input, output, self.activation_maps.get(name, Mt))

        for name, module in self.resnet.named_modules():
            #if isinstance(module, nn.Conv2d) and counter%frequency == 0:
            if name in self.layer_list:
                print(f"Attaching forward hook to layer: {name}")
                handle = module.register_forward_hook(create_hook_function(name, target, Mt))
                self.handles.append(handle)
                counter += 1
    
    def record_activation(self, module, output, layer_name):
        print(f"module: {module}")
        print(f"Recording activation for layer: {layer_name}")
        self.activation_maps[layer_name] = self.rec.forward_hook(module, None, output)
    
    def detach_forward_hook(self):
        # print('Detaching forward hook')
        for handle in self.handles:
            handle.remove()
        self.handles = []
        
    def forward(self, x, test=True, target=False):
        if not test:
            self.attach_forward_hook(target=target)
        
        x = self.resnet(x)
        
        if not test:
            self.detach_forward_hook()
        return x
    