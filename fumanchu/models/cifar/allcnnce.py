import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['allcnnce']


class AuxiliaryClassifier(nn.Module):

    def __init__(self, inplanes, num_classes, filters):
        super(AuxiliaryClassifier, self).__init__()
        self.inplanes = inplanes
        self.num_classes = num_classes
        self.stack = nn.Sequential(
            nn.Conv2d(self.inplanes, filters, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters, self.num_classes, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(6, self.num_classes),
        )

    def forward(self, x):
        return self.stack(x)


class AllCNNCE(nn.Module):

    def __init__(self, num_classes=10, dropRate=0.5, auxiliary_classifiers=False):
        super(AllCNNCE, self).__init__()
        channel_scale = 2.0 #4.0 / 3.0
        depth_scale = 1.0
        resolution_scale = 1.0

        self.auxiliary_classifiers = auxiliary_classifiers
        self.features1 = nn.Sequential(
            nn.Conv2d(3, int(96*channel_scale), kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(96*channel_scale), int(96*channel_scale), kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            # nn.Conv2d(int(96*channel_scale), int(96*channel_scale), kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            
            nn.Conv2d(int(96*channel_scale), int(96*channel_scale), kernel_size=3, stride=2, padding=1),
            nn.Dropout(p=dropRate)
        )
        self.features2 = nn.Sequential(
            nn.Conv2d(int(96*channel_scale), int(192*channel_scale), kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(192*channel_scale), int(192*channel_scale), kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            # nn.Conv2d(int(192*channel_scale), int(192*channel_scale), kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),

            nn.Conv2d(int(192*channel_scale), int(192*channel_scale), kernel_size=3, stride=2, padding=1),
            nn.Dropout(p=dropRate),
        )
        self.features3 = nn.Sequential(
            nn.Conv2d(int(192*channel_scale), int(192*channel_scale), kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(192*channel_scale), int(192*channel_scale), kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),

            # nn.Conv2d(int(192*channel_scale), int(192*channel_scale), kernel_size=1, stride=1, padding=0),
            # nn.ReLU(inplace=True),

            nn.Conv2d(int(192*channel_scale), num_classes, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(6, num_classes),
        )
        if self.auxiliary_classifiers:
            self.aux_branches = nn.ModuleList([
                AuxiliaryClassifier(int(96*channel_scale), num_classes, int(96*channel_scale)),
                AuxiliaryClassifier(int(192*channel_scale), num_classes, int(96*channel_scale)),
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


def allcnnce(**kwargs):
    r"""All-CNN-C model architecture from
    https://arxiv.org/pdf/1412.6806.pdf
    """
    model = AllCNNCE(**kwargs)
    return model
