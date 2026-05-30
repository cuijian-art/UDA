import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class DaNN(nn.Module):
    def __init__(self, n_class=31):
        super(DaNN, self).__init__()
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

        for param in list(self.vit.parameters())[:-20]:
            param.requires_grad = False

        self.label_predictor = nn.Linear(768, n_class)
        self.domain_discriminator = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def extract_feature(self, x):
        x = self.vit._process_input(x)
        n = x.shape[0]

        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.vit.encoder(x)

        feature = x[:, 0]
        return feature

    def forward(self, x, alpha=1.0):
        feature = self.extract_feature(x)

        label_pred = self.label_predictor(feature)

        reverse_feature = GradientReversalLayer.apply(feature, alpha)
        domain_pred = self.domain_discriminator(reverse_feature)

        return label_pred, domain_pred