import torch
import torch.nn as nn
import torchvision.models as models

class VGGPerceptualLoss(nn.Module):
    def __init__(self, 
                 resize=True, 
                 layer_ids=(3, 8, 15, 22),  
                 use_l1=True):
        super(VGGPerceptualLoss, self).__init__()
        
        vgg = models.vgg16(pretrained=True).features
        
        for param in vgg.parameters():
            param.requires_grad = False
   
        self.vgg_layers = nn.Sequential(*[vgg[i] for i in range(max(layer_ids) + 1)])
        
        self.layer_ids = layer_ids
        self.resize = resize
        self.criterion = nn.L1Loss() if use_l1 else nn.MSELoss()

    def forward(self, x, target):
        if self.resize:
            x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            target = nn.functional.interpolate(target, size=(224, 224), mode='bilinear', align_corners=False)

        
        x_features = []
        target_features = []

        current_x = x
        current_t = target

        for i, layer in enumerate(self.vgg_layers):
            current_x = layer(current_x)
            current_t = layer(current_t)
            if i in self.layer_ids:
                x_features.append(current_x)
                target_features.append(current_t)

        loss = 0
        for xf, tf in zip(x_features, target_features):
            loss += self.criterion(xf, tf)

        return loss
