import torch
import torch.nn as nn
import geffnet
import torchvision.models as models


sigmoid = torch.nn.Sigmoid()


# Gradient reversal class
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * 0.1


# Gradient reversal function
def grad_reverse(x):
    return GradReverse.apply(x)


# EfficientNet feature extractor
class enetv2(nn.Module):
    def __init__(self, enet_type, load_pretrained=True):
        super(enetv2, self).__init__()
        self.enet = geffnet.create_model(enet_type.replace('-', '_'), pretrained=True)
        self.dropout = nn.Dropout(0.5)
        in_ch = self.enet.classifier.in_features
        self.enet.classifier = nn.Identity()

    def extract(self, x):
        x = self.enet(x)
        return x

    def forward(self, x):
        # Assigning feature representation to new variable to allow it to be pulled out and passed into auxiliary head
        feat_out = self.extract(x).squeeze(-1).squeeze(-1)
        return feat_out


# ResNet-101 feature extractor
class ResNet101(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet101, self).__init__()
        self.enet = models.resnet101(pretrained=pretrained)
        self.dropouts = nn.Dropout(0.5)
        in_ch = self.enet.fc.in_features
        self.enet.fc = nn.Identity()

    def extract(self, x):
        x = self.enet(x)
        return x

    def forward(self, x):
        # Assigning feature representation to new variable to allow it to be pulled out and passed into auxiliary head
        feat_out = self.extract(x).squeeze(-1).squeeze(-1)
        return feat_out


# ResNeXt-101 feature extractor
class ResNext101(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNext101, self).__init__()
        self.enet = models.resnext101_32x8d(pretrained=pretrained)
        self.dropouts = nn.Dropout(0.5)
        in_ch = self.enet.fc.in_features
        self.enet.fc = nn.Identity()

    def extract(self, x):
        x = self.enet(x)
        return x

    def forward(self, x):
        # Assigning feature representation to new variable to allow it to be pulled out and passed into auxiliary head
        feat_out = self.extract(x).squeeze(-1).squeeze(-1)
        return feat_out


# Densenet feature extractor
class DenseNet(nn.Module):
    def __init__(self, pretrained=True):
        super(DenseNet, self).__init__()
        self.enet = models.densenet161(pretrained=pretrained)
        self.dropouts = nn.Dropout(0.5)
        in_ch = self.enet.classifier.in_features
        self.enet.classifier = nn.Identity()

    def extract(self, x):
        x = self.enet(x)
        return x

    def forward(self, x):
        # Assigning feature representation to new variable to allow it to be pulled out and passed into auxiliary head
        feat_out = self.extract(x).squeeze(-1).squeeze(-1)
        return feat_out


# Inception-V3 feature extractor
class Inception(nn.Module):
    def __init__(self, pretrained=True):
        super(Inception, self).__init__()
        self.enet = models.inception_v3(pretrained=pretrained)
        self.enet.aux_logits = False
        self.dropouts = nn.Dropout(0.5)
        in_ch = self.enet.fc.in_features
        self.enet.fc = nn.Identity()

    def extract(self, x):
        x = self.enet(x)
        return x

    def forward(self, x):
        # Assigning feature representation to new variable to allow it to be pulled out and passed into auxiliary head
        feat_out = self.extract(x).squeeze(-1).squeeze(-1)
        return feat_out


# Main classification head
class ClassificationHead(nn.Module):
    # Define model elements
    def __init__(self, out_dim, in_ch=1536):
        super(ClassificationHead, self).__init__()
        self.layer = nn.Linear(in_ch, out_dim)
        # Softmax function
        self.activation = nn.Softmax(dim=1)  # .Sigmoid()
        self.dropout = nn.Dropout(0.5)

    # Forward propagate input
    def forward(self, feat_out):
        # Feature map passed into fully connected layer to get logits
        x = self.layer(self.dropout(feat_out))  # .squeeze()
        # Returning logits
        return x


# Auxiliary head
class AuxiliaryHead(nn.Module):
    # Define model elements
    def __init__(self, num_aux, in_ch=1536):
        super(AuxiliaryHead, self).__init__()
        # Fully connected layer
        self.layer = nn.Linear(in_ch, num_aux)
        # Softmax function
        self.activation = nn.Softmax(dim=1)  # .Sigmoid()

    # Forward propagate input
    def forward(self, x_aux):
        # Feature map passed into fully connected layer to get logits
        x_aux = self.layer(x_aux).squeeze()
        # Probabilities output by using sigmoid activation
        px_aux = self.activation(x_aux)
        # Returning logits and probabilities as tuple
        return x_aux, px_aux


# Deeper auxiliary head (added fully connected layer)
class AuxiliaryHead2(nn.Module):
    # Define model elements
    def __init__(self, num_aux, in_ch=1536):
        super(AuxiliaryHead2, self).__init__()
        # Fully connected layer with 2 units
        self.layer = nn.Sequential(
               nn.Linear(in_ch, 128),
               nn.ReLU(),
               nn.Linear(128, num_aux))
        # Softmax function
        self.activation = nn.Softmax(dim=1)  # .Sigmoid()

    # Forward propagate input
    def forward(self, x_aux):
        # Feature map passed into fully connected layer to get logits
        x_aux = self.layer(x_aux).squeeze()
        # Probabilities output by using sigmoid activation
        px_aux = self.activation(x_aux)
        # Returning logits and probabilities as tuple
        return x_aux, px_aux
