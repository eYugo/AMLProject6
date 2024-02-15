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

class CASLayer(nn.Module):
    def __init__(self, K):
        super(CASLayer, self).__init__()
        self.topK = K
        
    def forward_hook(self, module, input, output, Mt=None, extension=0):
        A = output.clone().detach()
        torch.manual_seed(42)
        M = Mt.clone().detach() if Mt is not None else torch.bernoulli(torch.full_like(A, self.topK, device=A.device))
        if extension == 0:
            A_bin = torch.where(A <= 0, torch.tensor(0.0, device=A.device), torch.tensor(1.0, device=A.device))
            M_bin = torch.where(M <= 0, torch.tensor(0.0, device=M.device), torch.tensor(1.0, device=M.device))
            return A_bin * M_bin
        elif extension == 1:
            A_semi_bin = torch.where(A <= 0, torch.tensor(0.0, device=A.device), A)
            M_semi_bin = torch.where(M <= 0, torch.tensor(0.0, device=M.device), M)
            return A_semi_bin * M_semi_bin
        elif extension == 2:
            topK = int(A.flatten().size(0) * self.topK)
            values, indices = A.flatten().topk(topK, sorted=True)
            A_bin = torch.where(A <= 0, torch.tensor(0.0, device=A.device), A)
            M_bin = torch.where(M <= 0, torch.tensor(0.0, device=M.device), torch.tensor(1.0, device=M.device))
            mask = torch.zeros_like(A.flatten())
            mask[indices] = 1
            mask = mask.reshape_as(A)
            A_filtered = A_bin * mask
            # lower_top_k = values[-1]
            # M_bin = torch.where(A <= lower_top_k, torch.tensor(0.0, device=A.device), torch.tensor(1.0, device=A.device))
            
            return A_filtered * M_bin
        else:
            raise ValueError("Invalid extension value")

class RecordASLayer(nn.Module):
    def __init__(self):
        super(RecordASLayer, self).__init__()
    
    def forward_hook(self, module, input, output):
        A = output.clone().detach()
        return A

class ASHResNet18(nn.Module):
    def __init__(self, topK, layer_list=[], extension=0):
        super(ASHResNet18, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)
        self.mode = ''
        self.topK = topK
        self.cas = CASLayer(self.topK)
        self.rec = RecordASLayer()
        self.activation_maps = {}
        self.handles = []
        self.extension = extension
        self.layer_list = layer_list

    def attach_forward_hook(self):
        def create_hook_function(name):
            if self.mode=='record':
                # Targeting recording of activation maps
                return lambda module, input, output: self.record_activation(module, output, name)
            elif self.mode=='apply':
                # Targeting application of custom activation shaping
                return lambda module, input, output: self.cas.forward_hook(
                    module, input, output, self.activation_maps.get(name, None), extension=self.extension
                )

        for name, module in self.resnet.named_modules():
            if name in self.layer_list:
            #if name == "layer4.0.downsample.1":
                handle = module.register_forward_hook(create_hook_function(name))
                self.handles.append(handle)
        
    def record_activation(self, module, output, layer_name):
        self.activation_maps[layer_name] = self.rec.forward_hook(module, None, output)
    
    def detach_forward_hook(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        if self.mode != 'record':
            self.activation_maps = {}
        self.set_mode('')
    
    def set_mode(self, mode):
        self.mode = mode
    
    def forward(self, x, test=True):
        if not test:
            self.attach_forward_hook()
        x = self.resnet(x)
        
        if not test:
            self.detach_forward_hook()
        return x
    
