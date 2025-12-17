import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights
from .base import ModelWrapper

class VGGWrapper(ModelWrapper):
    
    def __init__(self, input_shape=(3, 224, 224), pretrained=True):
        self.input_shape = input_shape
        if pretrained:
            self.model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        else:
            self.model = vgg16(weights=None)
        
        self.model.eval()
    
    def get_model(self) -> nn.Module:
        return self.model
    
    def get_example_input(self, batch_size, device):
        return torch.randn(batch_size, *self.input_shape, device=device)
    
    def get_name(self) -> str:
        return "vgg16"

