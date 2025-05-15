import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.nn.functional import interpolate

import utils
from utils import compose_tensor


class TransformerNet(torch.nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        self.model = nn.Sequential(
            ConvBlock(3, 32, kernel_size=9, stride=1),
            ConvBlock(32, 64, kernel_size=3, stride=2),
            ConvBlock(64, 128, kernel_size=3, stride=2),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ConvBlock(128, 64, kernel_size=3, upsample=True),
            ConvBlock(64, 32, kernel_size=3, upsample=True),
            ConvBlock(32, 3, kernel_size=9, stride=1, normalize=False, relu=False),
        )

    def forward(self, x):
        return self.model(x)


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, stride=1, normalize=True, relu=True),
            ConvBlock(channels, channels, kernel_size=3, stride=1, normalize=True, relu=False),
        )

    def forward(self, x):
        return self.block(x) + x


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, upsample=False, normalize=True, relu=True):
        super(ConvBlock, self).__init__()
        self.upsample = upsample
        self.block = nn.Sequential(
            nn.ReflectionPad2d(kernel_size // 2), nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        )
        self.norm = nn.InstanceNorm2d(out_channels, affine=True) if normalize else None
        self.relu = relu

    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2)
        x = self.block(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.relu:
            x = F.relu(x)
        return x


class TestGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        pre_model = [nn.ReflectionPad2d(3),
                     nn.Conv2d(3, 32, 7),
                     nn.LeakyReLU(inplace=True),
                     nn.ReflectionPad2d(3),
                     nn.Conv2d(32, 32, 7),
                     nn.LeakyReLU(inplace=True),
                     nn.ReflectionPad2d(2),
                     nn.Conv2d(32, 32, 5),
                     nn.LeakyReLU(inplace=True),
                     nn.ReflectionPad2d(2),
                     nn.Conv2d(32, 32, 5),
                     nn.LeakyReLU(inplace=True),
                     nn.ReflectionPad2d(1),
                     nn.Conv2d(32, 3, 3),
                     nn.LeakyReLU(inplace=True)
                     ]
        self.pre_model = nn.Sequential(*pre_model)
        self.model = nn.Sequential(
            ConvBlock(3, 32, kernel_size=9, normalize=False, stride=1),
            ConvBlock(32, 64, kernel_size=3, normalize=False, stride=2),
            ConvBlock(64, 128, kernel_size=3, normalize=False, stride=2),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ConvBlock(128, 64, kernel_size=3, normalize=False, upsample=True),
            ConvBlock(64, 32, kernel_size=3, normalize=False, upsample=True),
            ConvBlock(32, 3, kernel_size=9, stride=1, normalize=False, relu=False),
        )

    def forward(self, input_tensor):
        out_tensor = input_tensor.detach().clone()
        max_tensor, max_indi = torch.max(input_tensor, dim=1)
        median_tensor, median_indi = torch.median(input_tensor, dim=1)
        median_tensor = torch.unsqueeze(median_tensor, dim=1)
        min_tensor, min_indi = torch.min(input_tensor, dim=1)
        max_tensor = torch.unsqueeze(max_tensor, dim=1)
        min_tensor = torch.unsqueeze(min_tensor, dim=1)
        max_indi = torch.unsqueeze(max_indi, dim=1)
        min_indi = torch.unsqueeze(min_indi, dim=1)
        med_indi = 3 - max_indi - min_indi
        med_indi = torch.clip(med_indi, 0, 2)
        V = max_tensor
        S = max_tensor - min_tensor
        H = max_tensor - median_tensor
        new_tensor = torch.concat((H, S, V), dim=1)
        y = self.pre_model(new_tensor)
        new_b, new_g, new_r = torch.split(y, split_size_or_sections=1, dim=1)
        out_tensor.scatter_(1, index=max_indi, src=new_b)
        out_tensor.scatter_(1, index=med_indi, src=new_g)
        out_tensor.scatter_(1, index=min_indi, src=new_r)

        out = self.model(out_tensor)
        return out


class ResBlock(nn.Module):
    def __init__(self, num_channels=128):
        super().__init__()
        model = [nn.ReflectionPad2d(1),
                 nn.Conv2d(num_channels, num_channels, 3),
                 nn.LeakyReLU(inplace=True),
                 nn.InstanceNorm2d(num_channels),
                 nn.ReflectionPad2d(1),
                 nn.Conv2d(num_channels, num_channels, 3)
                 ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return x + self.model(x)


class NoHSVModel(nn.Module):
    def __init__(self, in_channels=3, res_num=5):
        super().__init__()
        model0 = [nn.ReflectionPad2d(1),
                  nn.Conv2d(in_channels, 32, 3),
                  nn.LeakyReLU(inplace=True),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(32, 32, 3),
                  nn.LeakyReLU(inplace=True)]
        self.model0 = nn.Sequential(*model0)

        model1 = [nn.ReflectionPad2d(1),
                  nn.Conv2d(32, 64, 3, stride=2),
                  nn.LeakyReLU(inplace=True),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(64, 64, 3),
                  nn.LeakyReLU(inplace=True)
                  ]
        self.model1 = nn.Sequential(*model1)

        model2 = [nn.ReflectionPad2d(1),
                  nn.Conv2d(64, 128, 3, stride=2),
                  nn.LeakyReLU(inplace=True),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(128, 128, 3),
                  nn.LeakyReLU(inplace=True)
                  ]
        self.model2 = nn.Sequential(*model2)

        res_model = []
        for i in range(res_num):
            res_model = res_model + [ResBlock(128)]

        self.res_model = nn.Sequential(*res_model)

        model_trans2 = [nn.ReflectionPad2d(1),
                        nn.Conv2d(128, 64, 3),
                        nn.LeakyReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(64, 64, 3),
                        nn.LeakyReLU(inplace=True)]
        self.model_trans2 = nn.Sequential(*model_trans2)

        model_trans1 = [nn.ReflectionPad2d(1),
                        nn.Conv2d(64, 64, 3),
                        nn.LeakyReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(64, 32, 3),
                        nn.LeakyReLU(inplace=True)]
        self.model_trans1 = nn.Sequential(*model_trans1)

        model_trans0 = [nn.ReflectionPad2d(1),
                        nn.Conv2d(32, 32, 3),
                        nn.LeakyReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(32, 3, 3),
                        nn.LeakyReLU(inplace=True)]

        self.model_trans0 = nn.Sequential(*model_trans0)

    def forward(self, input_tensor):
        out_tensor = compose_tensor(input_tensor)
        x0 = self.model0(out_tensor)
        x1 = self.model1(x0)

        x2 = self.model2(x1)
        x2 = self.res_model(x2)

        x2 = F.interpolate(x2, size=(x1.shape[2], x1.shape[3]), mode='bilinear')
        x2 = self.model_trans2(x2)
        x3 = x1 + x2
        x3 = self.model_trans1(x3)
        x3 = F.interpolate(x3, size=(x0.shape[2], x0.shape[3]), mode='bilinear')
        x4 = x0 + x3
        return self.model_trans0(x4)


class GenNet(nn.Module):
    def __init__(self, in_channels=3, res_num=5):
        super().__init__()
        pre_model = [nn.ReflectionPad2d(1),
                     nn.Conv2d(in_channels, 32, 3),
                     nn.LeakyReLU(inplace=True),
                     nn.ReflectionPad2d(1),
                     nn.Conv2d(32, 32, 3),
                     nn.LeakyReLU(inplace=True),
                     nn.ReflectionPad2d(1),
                     nn.Conv2d(32, 3, 3),
                     nn.LeakyReLU(inplace=True)]
        self.pre_model = nn.Sequential(*pre_model)

        model0 = [nn.ReflectionPad2d(2),
                  nn.Conv2d(in_channels, 32, 5),
                  nn.LeakyReLU(inplace=True),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(32, 32, 3),
                  nn.LeakyReLU(inplace=True)]
        self.model0 = nn.Sequential(*model0)

        model1 = [nn.ReflectionPad2d(1),
                  nn.Conv2d(32, 64, 3, stride=2),
                  nn.LeakyReLU(inplace=True),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(64, 64, 3),
                  nn.LeakyReLU(inplace=True)
                  ]
        self.model1 = nn.Sequential(*model1)

        model2 = [nn.ReflectionPad2d(1),
                  nn.Conv2d(64, 128, 3, stride=2),
                  nn.LeakyReLU(inplace=True),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(128, 128, 3),
                  nn.LeakyReLU(inplace=True)
                  ]
        self.model2 = nn.Sequential(*model2)

        res_model = []
        for i in range(res_num):
            res_model = res_model + [ResBlock(128)]

        self.res_model = nn.Sequential(*res_model)

        model_trans2 = [nn.ReflectionPad2d(1),
                        nn.Conv2d(128, 64, 3),
                        nn.LeakyReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(64, 64, 3),
                        nn.LeakyReLU(inplace=True)]
        self.model_trans2 = nn.Sequential(*model_trans2)

        model_trans1 = [nn.ReflectionPad2d(1),
                        nn.Conv2d(128, 64, 3),
                        nn.LeakyReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(64, 32, 3),
                        nn.LeakyReLU(inplace=True)]
        self.model_trans1 = nn.Sequential(*model_trans1)

        model_trans0 = [nn.ReflectionPad2d(1),
                        nn.Conv2d(64, 32, 3),
                        nn.LeakyReLU(inplace=True),
                        nn.ReflectionPad2d(2),
                        nn.Conv2d(32, 3, 5),
                        nn.LeakyReLU(inplace=True)]

        self.model_trans0 = nn.Sequential(*model_trans0)

    def forward(self, input_tensor):
        out_tensor = input_tensor.detach().clone()
        max_tensor, max_indi = torch.max(input_tensor, dim=1)
        median_tensor, median_indi = torch.median(input_tensor, dim=1)
        median_tensor = torch.unsqueeze(median_tensor, dim=1)
        min_tensor, min_indi = torch.min(input_tensor, dim=1)
        max_tensor = torch.unsqueeze(max_tensor, dim=1)
        min_tensor = torch.unsqueeze(min_tensor, dim=1)
        max_indi = torch.unsqueeze(max_indi, dim=1)
        min_indi = torch.unsqueeze(min_indi, dim=1)
        med_indi = 3 - max_indi - min_indi
        R = input_tensor[:, 0: 1, :, :]
        G = input_tensor[:, 1: 2, :, :]
        med_indi = torch.clip(med_indi, 0, 2)
        V = max_tensor
        S = max_tensor - min_tensor / (max_tensor + 1e-2)
        H = max_tensor - median_tensor + R + G
        new_tensor = torch.concat((H, S, V), dim=1)
        y = self.pre_model(new_tensor)
        new_b, new_g, new_r = torch.split(y, split_size_or_sections=1, dim=1)
        out_tensor.scatter_(1, index=max_indi, src=new_b)
        out_tensor.scatter_(1, index=med_indi, src=new_g)
        out_tensor.scatter_(1, index=min_indi, src=new_r)
        out_tensor = compose_tensor(out_tensor)
        x0 = self.model0(out_tensor)
        x1 = self.model1(x0)

        x2 = self.model2(x1)
        x2 = self.res_model(x2)

        x2 = F.interpolate(x2, size=(x1.shape[2], x1.shape[3]), mode='bilinear')
        x2 = self.model_trans2(x2)
        x3 = torch.cat((x2, x1), dim=1)
        x3 = self.model_trans1(x3)
        x3 = F.interpolate(x3, size=(x0.shape[2], x0.shape[3]), mode='bilinear')
        x4 = torch.cat((x3, x0), dim=1)
        x4 = self.model_trans0(x4)
        return x4


class HSVGenerator(nn.Module):
    def __init__(self, in_channels=3, res_num=5):
        super().__init__()
        pre_model = [nn.ReflectionPad2d(1),
                     nn.Conv2d(in_channels, 32, 3),
                     nn.LeakyReLU(inplace=True),
                     nn.ReflectionPad2d(1),
                     nn.Conv2d(32, 32, 3),
                     nn.LeakyReLU(inplace=True),
                     nn.ReflectionPad2d(1),
                     nn.Conv2d(32, 3, 3),
                     nn.LeakyReLU(inplace=True)]
        self.pre_model = nn.Sequential(*pre_model)

        model0 = [nn.ReflectionPad2d(1),
                  nn.Conv2d(in_channels, 32, 3),
                  nn.LeakyReLU(inplace=True),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(32, 32, 3),
                  nn.LeakyReLU(inplace=True)]
        self.model0 = nn.Sequential(*model0)

        model1 = [nn.ReflectionPad2d(1),
                  nn.Conv2d(32, 64, 3, stride=2),
                  nn.LeakyReLU(inplace=True),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(64, 64, 3),
                  nn.LeakyReLU(inplace=True)
                  ]
        self.model1 = nn.Sequential(*model1)

        model2 = [nn.ReflectionPad2d(1),
                  nn.Conv2d(64, 128, 3, stride=2),
                  nn.LeakyReLU(inplace=True),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(128, 128, 3),
                  nn.LeakyReLU(inplace=True)
                  ]
        self.model2 = nn.Sequential(*model2)

        res_model = []
        for i in range(res_num):
            res_model = res_model + [ResBlock(128)]

        self.res_model = nn.Sequential(*res_model)

        model_trans2 = [nn.ReflectionPad2d(1),
                        nn.Conv2d(128, 64, 3),
                        nn.LeakyReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(64, 64, 3),
                        nn.LeakyReLU(inplace=True)]
        self.model_trans2 = nn.Sequential(*model_trans2)

        model_trans1 = [nn.ReflectionPad2d(1),
                        nn.Conv2d(128, 64, 3),
                        nn.LeakyReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(64, 32, 3),
                        nn.LeakyReLU(inplace=True)]
        self.model_trans1 = nn.Sequential(*model_trans1)

        model_trans0 = [nn.ReflectionPad2d(1),
                        nn.Conv2d(64, 32, 3),
                        nn.LeakyReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(32, 3, 3),
                        nn.LeakyReLU(inplace=True)]

        self.model_trans0 = nn.Sequential(*model_trans0)

    def forward(self, input_tensor):
        c_tensor = compose_tensor(input_tensor)
        x0 = self.model0(c_tensor)
        x1 = self.model1(x0)

        x2 = self.model2(x1)
        x2 = self.res_model(x2)

        x2 = F.interpolate(x2, size=(x1.shape[2], x1.shape[3]), mode='bilinear')
        x2 = self.model_trans2(x2)
        x3 = torch.cat((x2, x1), dim=1)
        x3 = self.model_trans1(x3)
        x3 = F.interpolate(x3, size=(x0.shape[2], x0.shape[3]), mode='bilinear')
        x4 = torch.cat((x3, x0), dim=1)
        x4 = self.model_trans0(x4)
        x4 = utils.decompose_tensor(x4)
        out_tensor = x4.detach().clone()
        max_tensor, max_indi = torch.max(x4, dim=1)
        R = x4[:, 0: 1, : , :]
        G = x4[:, 1: 2, :, :]
        median_tensor, median_indi = torch.median(x4, dim=1)
        median_tensor = torch.unsqueeze(median_tensor, dim=1)
        min_tensor, min_indi = torch.min(x4, dim=1)
        max_tensor = torch.unsqueeze(max_tensor, dim=1)
        min_tensor = torch.unsqueeze(min_tensor, dim=1)
        max_indi = torch.unsqueeze(max_indi, dim=1)
        min_indi = torch.unsqueeze(min_indi, dim=1)
        median_indi = torch.unsqueeze(median_indi, dim=1)
        V = max_tensor
        S = max_tensor - min_tensor
        H = median_tensor - max_tensor + R + G

        new_tensor = torch.concat((H, S, V), dim=1)
        y = self.pre_model(new_tensor)
        new_b, new_g, new_r = torch.split(y, split_size_or_sections=1, dim=1)
        out_tensor.scatter_(1, index=max_indi, src=new_b)
        out_tensor.scatter_(1, index=median_indi, src=new_g)
        out_tensor.scatter_(1, index=min_indi, src=new_r)
        out_tensor = utils.compose_tensor(out_tensor)
        return out_tensor


class ColorDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.model = nn.Sequential(nn.ReflectionPad2d(1),
                                   spectral_norm(nn.Conv2d(in_channels, 18, 3, groups=1)),
                                   nn.LeakyReLU(0.2),
                                   nn.ReflectionPad2d(1),
                                   spectral_norm(nn.Conv2d(18, 18, 3, groups=1, stride=2)),
                                   nn.LeakyReLU(0.2),
                                   )

        self.model1 = nn.Sequential(nn.ReflectionPad2d(1),
                                    spectral_norm(nn.Conv2d(18, 36, 3, groups=1)),
                                    nn.LeakyReLU(0.2),
                                    nn.ReflectionPad2d(1),
                                    spectral_norm(nn.Conv2d(36, 36, 3, groups=1, stride=2)),
                                    nn.LeakyReLU(0.2),
                                    )
        self.model2 = nn.Sequential(nn.ReflectionPad2d(1),
                                    spectral_norm(nn.Conv2d(36, 72, 3, groups=1)),
                                    nn.LeakyReLU(0.2),
                                    nn.ReflectionPad2d(1),
                                    spectral_norm(nn.Conv2d(72, 72, 3, groups=1, stride=2)),
                                    nn.LeakyReLU(0.2)
                                    )
        self.model3 = nn.Sequential(nn.ReflectionPad2d(1),
                                    spectral_norm(nn.Conv2d(72, 144, 3, groups=1)),
                                    nn.LeakyReLU(0.2),
                                    nn.ReflectionPad2d(1),
                                    spectral_norm(nn.Conv2d(144, 144, 3, groups=1, stride=2)),
                                    nn.LeakyReLU(0.2)
                                    )
        self.model4 = nn.Sequential(nn.ReflectionPad2d(1),
                                    spectral_norm(nn.Conv2d(144, 64, 3)),
                                    nn.LeakyReLU(0.2),
                                    nn.ReflectionPad2d(1),
                                    spectral_norm(nn.Conv2d(64, 3, 3, stride=2)),
                                    nn.LeakyReLU(0.2),
                                    )

    def forward(self, input_tensor):
        max_tensor, max_indi = torch.max(input_tensor, dim=1)
        median_tensor, median_indi = torch.median(input_tensor, dim=1)
        median_tensor = torch.unsqueeze(median_tensor, dim=1)
        min_tensor, min_indi = torch.min(input_tensor, dim=1)
        max_tensor = torch.unsqueeze(max_tensor, dim=1)
        min_tensor = torch.unsqueeze(min_tensor, dim=1)
        R = input_tensor[:, 0: 1, :, :]
        G = input_tensor[:, 1: 2, :, :]
        V = max_tensor
        S = (max_tensor - min_tensor) / (torch.abs(max_tensor) + 1e-2)
        H = median_tensor - min_tensor + R + G

        new_tensor = torch.concat((H, S, V), dim=1)
        x1 = self.model(new_tensor)
        x1 = self.model1(x1)
        x1 = self.model2(x1)
        x1 = self.model3(x1)
        x1 = self.model4(x1)
        x1 = nn.Sigmoid()(x1)
        return torch.mean(x1, dim=(1, 2, 3))


class ColorDis(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.model1 = nn.Sequential(nn.ReflectionPad2d(1),
                                    nn.Conv2d(in_channels, 18, 3, groups=1),
                                    nn.LeakyReLU(inplace=True),
                                    nn.ReflectionPad2d(1),
                                    nn.Conv2d(18, 36, 3, groups=1),
                                    nn.LeakyReLU(inplace=True),
                                    nn.ReflectionPad2d(1),
                                    spectral_norm(nn.Conv2d(36, 64, 3)),
                                    nn.LeakyReLU(inplace=True),
                                    nn.ReflectionPad2d(1),
                                    spectral_norm(nn.Conv2d(64, 32, 3)),
                                    nn.LeakyReLU(inplace=True),
                                    nn.ReflectionPad2d(1),
                                    spectral_norm(nn.Conv2d(32, 32, 1)),
                                    nn.LeakyReLU(inplace=True),
                                    nn.ReflectionPad2d(1),
                                    spectral_norm(nn.Conv2d(32, 32, 1)),
                                    nn.LeakyReLU(inplace=True),
                                    nn.ReflectionPad2d(1),
                                    spectral_norm(nn.Conv2d(32, 32, 1)),
                                    nn.LeakyReLU(inplace=True),
                                    nn.Conv2d(32, 32, 1),
                                    )
        self.model2 = nn.Sequential(nn.ReflectionPad2d(1),
                                    nn.Conv2d(32, 128, 1))

    def forward(self, input_tensor):
        max_tensor, max_indi = torch.max(input_tensor, dim=1)
        median_tensor, median_indi = torch.median(input_tensor, dim=1)
        median_tensor = torch.unsqueeze(median_tensor, dim=1)
        min_tensor, min_indi = torch.min(input_tensor, dim=1)
        max_tensor = torch.unsqueeze(max_tensor, dim=1)
        min_tensor = torch.unsqueeze(min_tensor, dim=1)
        V = max_tensor
        S = (max_tensor - min_tensor) / (max_tensor + 1e-2)
        H = max_tensor - median_tensor
        new_tensor = torch.concat((H, S, V), dim=1)
        y = self.model1(new_tensor)
        y = -torch.abs(y)
        y = self.model2(y)
        y = torch.clip(y, 0, 1)
        return torch.mean(y, dim=(1, 2, 3))


class Dis(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.model1 = nn.Sequential(
                                    nn.Conv2d(in_channels, 300, 1, groups=1),
                                    )
        self.model2 = nn.Sequential(nn.LeakyReLU(inplace=True),
                                    nn.Conv2d(300, 1, 1))

    def forward(self, input_tensor):
        max_tensor, max_indi = torch.max(input_tensor, dim=1)
        median_tensor, median_indi = torch.median(input_tensor, dim=1)
        median_tensor = torch.unsqueeze(median_tensor, dim=1)
        min_tensor, min_indi = torch.min(input_tensor, dim=1)
        max_tensor = torch.unsqueeze(max_tensor, dim=1)
        min_tensor = torch.unsqueeze(min_tensor, dim=1)
        V = max_tensor
        S = max_tensor - min_tensor
        H = max_tensor - median_tensor
        new_tensor = torch.concat((H, S, V), dim=1)
        y = self.model1(new_tensor)
        y = - torch.abs(y)
        y = self.model2(y)
        y = nn.Sigmoid()(y)
        return torch.mean(y, dim=[1, 2, 3])


class GramDiscriminator(nn.Module):
    def __init__(self, in_channels1=1, in_channels2=1, in_channels3=1, in_channels4=1):
        super().__init__()
        model1 = [nn.ReflectionPad2d(2),
                  spectral_norm(nn.Conv2d(in_channels1, 32, 5)),
                  nn.LeakyReLU(0.2, inplace=True),
                  nn.ReflectionPad2d(1),
                  spectral_norm(nn.Conv2d(32, 32, 3)),
                  nn.LeakyReLU(0.2, inplace=True),
                  nn.ReflectionPad2d(1),
                  spectral_norm(nn.Conv2d(32, in_channels1, 3, 2)),
                  nn.LeakyReLU(0.2, inplace=True)
                  ]

        model2 = [nn.ReflectionPad2d(1),
                  spectral_norm(nn.Conv2d(in_channels2 * 2, 64, 3)),
                  nn.LeakyReLU(0.2, inplace=True),
                  nn.ReflectionPad2d(1),
                  spectral_norm(nn.Conv2d(64, 64, 3)),
                  nn.LeakyReLU(0.2, inplace=True),
                  nn.ReflectionPad2d(1),
                  spectral_norm(nn.Conv2d(64, in_channels2, 3, 2)),
                  nn.LeakyReLU(0.2, inplace=True)
                  ]

        model3 = [nn.ReflectionPad2d(1),
                  spectral_norm(nn.Conv2d(in_channels3 * 3, 128, 3)),
                  nn.LeakyReLU(0.2, inplace=True),
                  nn.ReflectionPad2d(1),
                  spectral_norm(nn.Conv2d(128, 128, 3)),
                  nn.LeakyReLU(0.2, inplace=True),
                  nn.ReflectionPad2d(1),
                  spectral_norm(nn.Conv2d(128, in_channels3, 3, 2)),
                  nn.LeakyReLU(0.2, inplace=True)
                  ]

        model4 = [nn.ReflectionPad2d(1),
                  spectral_norm(nn.Conv2d(in_channels4 * 4, 256, 3)),
                  nn.LeakyReLU(0.2, inplace=True),
                  nn.ReflectionPad2d(1),
                  spectral_norm(nn.Conv2d(256, 256, 3)),
                  nn.LeakyReLU(0.2, inplace=True),
                  nn.ReflectionPad2d(1),
                  spectral_norm(nn.Conv2d(256, in_channels1 * 5, 3, 2)),
                  nn.LeakyReLU(0.2, inplace=True)
                  ]

        model = [nn.ReflectionPad2d(1),
                 spectral_norm(nn.Conv2d(in_channels1 * 5, 256, 3, 2)),
                 nn.LeakyReLU(0.2, inplace=True),
                 nn.ReflectionPad2d(1),
                 spectral_norm(nn.Conv2d(256, 64, 3, 4)),
                 nn.LeakyReLU(0.2, inplace=True),
                 nn.ReflectionPad2d(1),
                 spectral_norm(nn.Conv2d(64, 1, 3, 4)),
                 nn.LeakyReLU(0.2, inplace=True),
                 ]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)

        self.model4 = nn.Sequential(*model4)
        self.model = nn.Sequential(*model)
        self.sigmoid = nn.Sigmoid()

    def forward(self, tensor_list):
        input_tensor1 = tensor_list[3]
        input_tensor2 = tensor_list[2]
        input_tensor3 = tensor_list[1]
        input_tensor4 = tensor_list[0]
        y1 = self.model1(input_tensor1)
        x1 = interpolate(input_tensor1, scale_factor=0.5)
        x1 = x1 + y1
        x2 = torch.cat((x1, input_tensor2), dim=1)
        y2 = self.model2(x2)
        x2 = interpolate(x2, scale_factor=0.5)
        x2 = x2 + y2
        x3 = torch.cat((x2, input_tensor3), dim=1)
        y3 = self.model3(x3)
        x3 = interpolate(x3, scale_factor=0.5)
        x3 = x3 + y3
        x4 = torch.cat((x3, input_tensor4), dim=1)
        y4 = self.model4(x4)
        y = self.model(y4).view(-1)
        return self.sigmoid(y)


class HSVGen(nn.Module):
    def __init__(self, in_channels=3, res_num=5):
        super().__init__()
        pre_model = [nn.ReflectionPad2d(1),
                     nn.Conv2d(in_channels, 32, 3),
                     nn.LeakyReLU(inplace=True),
                     nn.ReflectionPad2d(1),
                     nn.Conv2d(32, 32, 3),
                     nn.LeakyReLU(inplace=True),
                     nn.ReflectionPad2d(1),
                     nn.Conv2d(32, 3, 3),
                     nn.LeakyReLU(inplace=True)]
        self.pre_model = nn.Sequential(*pre_model)

        model0 = [nn.ReflectionPad2d(1),
                  nn.Conv2d(in_channels, 32, 3),
                  nn.LeakyReLU(inplace=True),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(32, 32, 3),
                  nn.LeakyReLU(inplace=True)]
        self.model0 = nn.Sequential(*model0)

        model1 = [nn.ReflectionPad2d(1),
                  nn.Conv2d(32, 64, 3, stride=2),
                  nn.LeakyReLU(inplace=True),
                  nn.InstanceNorm2d(64),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(64, 64, 3),
                  nn.LeakyReLU(inplace=True)
                  ]
        self.model1 = nn.Sequential(*model1)

        model2 = [nn.ReflectionPad2d(1),
                  nn.Conv2d(64, 128, 3, stride=2),
                  nn.LeakyReLU(inplace=True),
                  nn.InstanceNorm2d(128),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(128, 128, 3),
                  nn.LeakyReLU(inplace=True)
                  ]
        self.model2 = nn.Sequential(*model2)

        res_model = []
        for i in range(res_num):
            res_model = res_model + [ResBlock(128)]

        self.res_model = nn.Sequential(*res_model)

        model_trans2 = [nn.ReflectionPad2d(1),
                        nn.Conv2d(128, 64, 3),
                        nn.LeakyReLU(inplace=True),
                        nn.InstanceNorm2d(64),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(64, 64, 3),
                        nn.LeakyReLU(inplace=True)]
        self.model_trans2 = nn.Sequential(*model_trans2)

        model_trans1 = [nn.ReflectionPad2d(1),
                        nn.Conv2d(64, 64, 3),
                        nn.LeakyReLU(inplace=True),
                        nn.InstanceNorm2d(64),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(64, 32, 3),
                        nn.LeakyReLU(inplace=True)]
        self.model_trans1 = nn.Sequential(*model_trans1)

        model_trans0 = [nn.ReflectionPad2d(1),
                        nn.Conv2d(32, 32, 3),
                        nn.LeakyReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(32, 3, 3),
                        nn.LeakyReLU(inplace=True)]

        self.model_trans0 = nn.Sequential(*model_trans0)

    def forward(self, input_tensor):
        out_tensor = input_tensor.detach().clone()
        max_tensor, max_indi = torch.max(input_tensor, dim=1)
        R = input_tensor[:, 0: 1, : , :]
        G = input_tensor[:, 1: 2, :, :]
        median_tensor, median_indi = torch.median(input_tensor, dim=1)
        median_tensor = torch.unsqueeze(median_tensor, dim=1)
        min_tensor, min_indi = torch.min(input_tensor, dim=1)
        max_tensor = torch.unsqueeze(max_tensor, dim=1)
        min_tensor = torch.unsqueeze(min_tensor, dim=1)
        max_indi = torch.unsqueeze(max_indi, dim=1)
        min_indi = torch.unsqueeze(min_indi, dim=1)
        median_indi = torch.unsqueeze(median_indi, dim=1)
        V = max_tensor
        S = max_tensor - min_tensor / (max_tensor + 1e-2)
        H = median_tensor - max_tensor + R + G
        new_tensor = torch.concat((H, S, V), dim=1)
        y = self.pre_model(new_tensor)
        new_b, new_g, new_r = torch.split(y, split_size_or_sections=1, dim=1)
        out_tensor.scatter_(1, index=max_indi, src=new_b)
        out_tensor.scatter_(1, index=median_indi, src=new_g)
        out_tensor.scatter_(1, index=min_indi, src=new_r)
        out_tensor = utils.compose_tensor(out_tensor)
        c_tensor = compose_tensor(out_tensor)
        x0 = self.model0(c_tensor)
        x1 = self.model1(x0)

        x2 = self.model2(x1)
        x2 = self.res_model(x2)

        x2 = F.interpolate(x2, size=(x1.shape[2], x1.shape[3]), mode='bilinear')
        x2 = self.model_trans2(x2)
        x3 = x1 + x2
        x3 = self.model_trans1(x3)
        x3 = F.interpolate(x3, size=(x0.shape[2], x0.shape[3]), mode='bilinear')
        x4 = x3 + x0
        x4 = self.model_trans0(x4)
        return x4
