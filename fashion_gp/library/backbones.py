import torch
import torchvision.models as models

__all__ = ['ResnetModel']


class ResnetModel(torch.nn.Module):
    def __init__(self, num_labels, backbone, set_img_num=1, remove_last_layer=True,):
        super().__init__()

        if backbone == "resnet18":
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.latent_dim = 512
        elif backbone == "resnet50":
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.latent_dim = 2048
        elif backbone == "resnet101":
            self.backbone = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
            self.latent_dim = 2048
        else:
            raise NotImplementedError(f'Unknown backbone: {backbone!r}')
        if remove_last_layer:
            self.backbone.fc = torch.nn.Identity()
        else:
            self.latent_dim = 1000

        self.fc = torch.nn.Linear(self.latent_dim*set_img_num, num_labels)

    def forward(self, images):
        features = self.backbone(images)
        output = self.fc(features)
        # output = nn.functional.softmax(features, dim=1)
        return output

# class Vit(torch.nn.Module):
#     def __init__(self, num_labels=10, backbone=None):
#         super(Vit, self).__init__()
#         # self.fc = nn.Linear()
#         if backbone is not None:
#             self.model = ViTModel.from_pretrained(backbone)
#         else:
#             self.model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
#         self.num_labels = num_labels
#         self.classifier = nn.Linear(self.model.config.hidden_size, num_labels)
#
#     def forward(self,images):
#         output = self.model(images)
#         return self.classifier(output.last_hidden_state[:, 0])
