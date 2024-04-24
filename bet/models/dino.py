import torchvision
import torch.nn as nn
import torch
# import utils
from einops import rearrange
from transformers import ViTImageProcessor, ViTModel
import requests



class dino(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        freeze_pretrained: bool = True,
        output_dim: int = 64  # This might not apply directly as ViT models have different output features
    ):
        super().__init__()
        # Initialize the ViT model with pretrained weights
        self.feature_extractor = ViTImageProcessor.from_pretrained('facebook/dino-vitb8')
        self.model = ViTModel.from_pretrained('facebook/dino-vitb8')

        # Flatten the last hidden state output to get the pooled features
        self.flatten = nn.Flatten(start_dim=1)

        # Setting up model freezing if required
        if pretrained and freeze_pretrained:
            for param in self.model.parameters():
                param.requires_grad = False

        # Normalization is already part of ViTFeatureExtractor, no need to double-apply
        # This is kept if there's a specific need to re-apply similar ImageNet normalization
        self.normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def forward(self, x):
        # Check if input is a sequence and flatten it appropriately
        is_seq = x.dim() == 5
        if is_seq:
            n, t, c, h, w = x.shape
            x = rearrange(x, "n t c h w -> (n t) c h w")

        # Process images through the feature extractor
        inputs = self.feature_extractor(images=x, return_tensors="pt")
        outputs = self.model(**inputs)

        # Use the last hidden state; you could alternatively use 'pooler_output' if defined
        out = self.flatten(outputs.last_hidden_state)

        if is_seq:
            out = rearrange(out, "(n t) e -> n t e", n=n, t=t)

        return out

model = dino()
# Example input tensor (randomly generated for demonstration; replace with actual image tensor)
x = torch.rand(1, 3, 448, 448)  # Simulating one image of size 224x224 with 3 color channels
output = model(x)
print(output.shape)