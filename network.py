import torch.nn as nn
from torchvision import models
import torch

resnet_dict = {"ResNet18": models.resnet18, "ResNet34": models.resnet34,
               "ResNet50": models.resnet50, "ResNet101": models.resnet101,
               "ResNet152": models.resnet152}


class ResNet(nn.Module):
    def __init__(self, hash_bit, res_model="ResNet50"):
        super(ResNet, self).__init__()
        model_resnet = resnet_dict[res_model](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, self.layer1, self.layer2,
                                            self.layer3, self.layer4)
        self.hash_layer = nn.Linear(model_resnet.fc.in_features, hash_bit)
        self.hash_layer.weight.data.normal_(0, 0.01)
        self.hash_layer.bias.data.fill_(0.0)

    def forward(self, x):
        features = self.feature_layers(x)
        x = self.avgpool(features)
        x = x.view(x.size(0), -1)
        x = self.hash_layer(x)

        layer = nn.Tanh()
        advx = layer(x)
        return x, advx, features

    def forward_bn(self, x, device, bn_param):
        features = self.feature_layers(x)
        x = self.avgpool(features)
        x = x.view(x.size(0), -1)
        x = self.hash_layer(x)
        # print(x.max(), x.min(), x.shape[1])
        bn = nn.BatchNorm1d(x.shape[1]).to(device)
        x_bn = bn(x) * bn_param
        layer = nn.Tanh()
        advx = layer(x_bn)
        return x, advx, features


    def adv_forward(self, x, alpha=1):
        features = self.feature_layers(x)
        x = self.avgpool(features)
        x = x.view(x.size(0), -1)
        x = self.hash_layer(x)
        layer = nn.Tanh()
        y = layer(alpha * x)
        return x, y, features


# Vgg
vgg_dict = {"Vgg11": models.vgg11, "Vgg13": models.vgg13,
            "Vgg16": models.vgg16, "Vgg19": models.vgg19}


class Vgg(nn.Module):
    def __init__(self, hash_bit, vgg_model="Vgg16"):
        super(Vgg, self).__init__()
        model_vgg = vgg_dict[vgg_model](pretrained=True)
        self.features = model_vgg.features
        cl1 = nn.Linear(512 * 7 * 7, 4096)
        cl1.weight = model_vgg.classifier[0].weight
        cl1.bias = model_vgg.classifier[0].bias

        cl2 = nn.Linear(4096, 4096)
        cl2.weight = model_vgg.classifier[3].weight
        cl2.bias = model_vgg.classifier[3].bias

        self.hash_layer = nn.Sequential(
            cl1,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            cl2,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, hash_bit),
        )

    def forward(self, x):
        features = self.features(x)
        x = features.contiguous().view(features.size(0), 512 * 7 * 7)
        x = self.hash_layer(x)

        layer = nn.Tanh()
        advx = layer(x)
        return x, advx, features

    def forward_bn(self, x, device, bn_param):
        features = self.features(x)
        x = features.contiguous().view(features.size(0), 512 * 7 * 7)
        x = self.hash_layer(x)
        # print(x.max(), x.min(), x.shape[1])
        bn = nn.BatchNorm1d(x.shape[1]).to(device)
        x_bn = bn(x) * bn_param
        layer = nn.Tanh()
        advx = layer(x_bn)
        return x, advx, features

    def adv_forward(self, x, alpha=1):
        features = self.features(x)
        x = features.contiguous().view(features.size(0), 512 * 7 * 7)
        x = self.hash_layer(x)
        layer = nn.Tanh()
        y = layer(alpha * x)
        return x, y, features



### DHD algorithm
# ResNet

class ResNet_Robust(nn.Module):
    def __init__(self, resnet_model):
        super(ResNet_Robust, self).__init__()
        self.pretrained = resnet_dict[resnet_model](pretrained=True)
        self.children_list = []
        for n, c in self.pretrained.named_children():
            if n == 'avgpool':
                break
            self.children_list.append(c)

        self.avgpool = self.pretrained.avgpool
        self.net = nn.Sequential(*self.children_list)
        self.pretrained = None

    def forward(self, x):
        features = self.net(x)
        x = self.avgpool(features)
        x = torch.flatten(x, 1)
        return x, features


class Vgg_Robust(nn.Module):
    def __init__(self, vgg_model):
        super(Vgg_Robust, self).__init__()
        model_vgg = vgg_dict[vgg_model](pretrained=True)

        self.features = model_vgg.features
        cl1 = nn.Linear(512 * 7 * 7, 4096)
        cl1.weight = model_vgg.classifier[0].weight
        cl1.bias = model_vgg.classifier[0].bias

        cl2 = nn.Linear(4096, 4096)
        cl2.weight = model_vgg.classifier[3].weight
        cl2.bias = model_vgg.classifier[3].bias

        self.hash_layer = nn.Sequential(
            cl1,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            cl2,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            # nn.Linear(4096, hash_bit),        ###
        )

    def forward(self, x):
        features = self.features(x)  # 512,7,7
        x = features.contiguous().view(features.size(0), 512 * 7 * 7)
        x = self.hash_layer(x)
        x = torch.flatten(x, 1)
        return x, features


class Hash_func(nn.Module):
    def __init__(self, fc_dim, N_bits, NB_CLS):
        super(Hash_func, self).__init__()
        self.Hash = nn.Sequential(
            nn.Linear(fc_dim, N_bits, bias=False),
            nn.LayerNorm(N_bits))
        self.P = nn.Parameter(torch.FloatTensor(NB_CLS, N_bits), requires_grad=True)

    def forward(self, X):
        X_out, X_fea = X
        X_out = self.Hash(X_out)
        return X_out, torch.tanh(X_out), X_fea

    def adv_forward(self, X, alpha=1):
        X_out, X_fea = X
        X_out = self.Hash(X_out)
        return torch.tanh(alpha * X_out), X_fea

### DHD algorithm
# ResNet

class ResNet_Robust_adv(nn.Module):
    def __init__(self, resnet_model):
        super(ResNet_Robust_adv, self).__init__()
        self.pretrained = resnet_dict[resnet_model](pretrained=True)
        self.children_list = []
        for n, c in self.pretrained.named_children():
            if n == 'avgpool':
                break
            self.children_list.append(c)

        self.avgpool = self.pretrained.avgpool
        self.net = nn.Sequential(*self.children_list)
        self.pretrained = None

    def forward(self, input):
        x, alpha = input
        features = self.net(x)
        x = self.avgpool(features)
        x = torch.flatten(x, 1)
        return (x, features, alpha)


class Vgg_Robust_adv(nn.Module):
    def __init__(self, vgg_model):
        super(Vgg_Robust_adv, self).__init__()
        model_vgg = vgg_dict[vgg_model](pretrained=True)

        self.features = model_vgg.features
        cl1 = nn.Linear(512 * 7 * 7, 4096)
        cl1.weight = model_vgg.classifier[0].weight
        cl1.bias = model_vgg.classifier[0].bias

        cl2 = nn.Linear(4096, 4096)
        cl2.weight = model_vgg.classifier[3].weight
        cl2.bias = model_vgg.classifier[3].bias

        self.hash_layer = nn.Sequential(
            cl1,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            cl2,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            # nn.Linear(4096, hash_bit),        ###
        )

    def forward(self, input):
        x, alpha = input
        features = self.features(x)  # 512,7,7
        x = features.contiguous().view(features.size(0), 512 * 7 * 7)
        x = self.hash_layer(x)
        x = torch.flatten(x, 1)
        return (x, features, alpha)


class Hash_func_adv(nn.Module):
    def __init__(self, fc_dim, N_bits, NB_CLS):
        super(Hash_func_adv, self).__init__()
        self.Hash = nn.Sequential(
            nn.Linear(fc_dim, N_bits, bias=False),
            nn.LayerNorm(N_bits))
        self.P = nn.Parameter(torch.FloatTensor(NB_CLS, N_bits), requires_grad=True)

    def forward(self, X):
        X_out, X_fea, alpha = X
        X_out = self.Hash(X_out)
        return X_out, torch.tanh(alpha * X_out), X_fea