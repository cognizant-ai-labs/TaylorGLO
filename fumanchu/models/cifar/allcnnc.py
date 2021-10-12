import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['allcnnc']


class AuxiliaryClassifier(nn.Module):

    def __init__(self, inplanes, num_classes, activation_fn=nn.ReLU(inplace=True)):
        super(AuxiliaryClassifier, self).__init__()
        self.inplanes = inplanes
        self.num_classes = num_classes
        self.relu = activation_fn
        self.stack = nn.Sequential(
            nn.Conv2d(self.inplanes, 96, kernel_size=1, stride=1, padding=0),
            self.relu,
            nn.Conv2d(96, self.num_classes, kernel_size=1, stride=1, padding=0),
            self.relu,
            nn.AvgPool2d(6, self.num_classes),
        )

    def forward(self, x):
        return self.stack(x)


class AllCNNC(nn.Module):

    def __init__(self, num_classes=10, dropRate=0.5, auxiliary_classifiers=False, activation_fn=nn.ReLU(inplace=True)):
        super(AllCNNC, self).__init__()
        self.auxiliary_classifiers = auxiliary_classifiers
        self.relu = activation_fn
        self.features1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1),
            self.relu,
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
            self.relu,
            
            nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1),
            nn.Dropout(p=dropRate)
        )
        self.features2 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=3, stride=1, padding=1),
            self.relu,
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
            self.relu,

            nn.Conv2d(192, 192, kernel_size=3, stride=2, padding=1),
            nn.Dropout(p=dropRate),
        )
        self.features3 = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
            self.relu,
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
            self.relu,
            nn.Conv2d(192, num_classes, kernel_size=1, stride=1, padding=0),
            self.relu,
            nn.AvgPool2d(6, num_classes),
        )
        if self.auxiliary_classifiers:
            self.aux_branches = nn.ModuleList([
                AuxiliaryClassifier(96, num_classes),
                AuxiliaryClassifier(192, num_classes),
            ])

    def forward(self, x):   
        if self.auxiliary_classifiers:
            x = self.features1(x)
            aux_1_branch = self.aux_branches[0](x)
            aux_1 = aux_1_branch.view(aux_1_branch.size(0), -1)

            x = self.features2(x)
            aux_2_branch = self.aux_branches[1](x)
            aux_2 = aux_2_branch.view(aux_2_branch.size(0), -1)

            x = self.features3(x)
            x = x.view(x.size(0), -1)
            return x, aux_1, aux_2

        else:

            x = self.features1(x)
            x = self.features2(x)
            x = self.features3(x)
            x = x.view(x.size(0), -1)
            return x


def allcnnc(**kwargs):
    r"""All-CNN-C model architecture from
    https://arxiv.org/pdf/1412.6806.pdf
    """
    model = AllCNNC(**kwargs)
    return model
