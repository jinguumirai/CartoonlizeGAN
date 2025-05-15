import torch
import torch.nn as nn
from torch.nn.functional import interpolate
from torchvision.models import vgg16, vgg19
import torchvision
from torch.nn.functional import conv2d


class TVLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        input_shape = input_tensor.shape
        height = input_shape[2]
        width = input_shape[3]
        tv_loss1 = torch.mean((input_tensor[:, :, 1:, :] - input_tensor[:, :, :height - 1, :]) ** 2)
        tv_loss2 = torch.mean((input_tensor[:, :, :, 1:] - input_tensor[:, :, :, :width - 1]) ** 2)

        return (tv_loss2 + tv_loss1) / height / width / 3

class ITVLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor, edge_tensor):
        new_edge = torch.cat((edge_tensor, edge_tensor, edge_tensor), dim=0)
        input_shape = input_tensor.shape
        height = input_shape[2]
        width = input_shape[3]
        tv_loss1 = torch.abs((input_tensor[:, :, 1:, :] - input_tensor[:, :, :height - 1, :])) * (1 - new_edge[:, :, 1:, :])
        tv_loss2 = torch.abs((input_tensor[:, :, :, 1:] - input_tensor[:, :, :, :width - 1])) * (1 - new_edge[:, :, :, 1:])

        return (torch.mean(tv_loss1) + torch.mean(tv_loss2)) / 2

"""
class TorchVGG16(nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg16(pretrained=True).features
        conv1_2_list = []
        conv2_2_list = []
        conv3_3_list = []
        conv4_3_list = []
        conv5_3_list = []
        for x in range(4):
            conv1_2_list.append(vgg_pretrained_features[x])
        self.conv1_2 = torch.nn.Sequential(*conv1_2_list)
        for x in range(4, 9):
            conv2_2_list.append(vgg_pretrained_features[x])
        self.conv2_2 = torch.nn.Sequential(*conv2_2_list)
        for x in range(9, 16):
            conv3_3_list.append(vgg_pretrained_features[x])
        self.conv3_3 = torch.nn.Sequential(*conv3_3_list)
        for x in range(16, 23):
            conv4_3_list.append(vgg_pretrained_features[x])
        self.conv4_3 = torch.nn.Sequential(*conv4_3_list)
        for x in range(23, 26):
            conv5_3_list.append(vgg_pretrained_features[x])
        self.conv5_3 = torch.nn.Sequential(*conv5_3_list)
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        relu1_2 = self.conv1_2(X)
        relu2_2 = self.conv2_2(relu1_2)
        relu3_3 = self.conv3_3(relu2_2)
        relu4_3 = self.conv4_3(relu3_3)
        relu5_2 = self.conv5_3(relu4_3)
        return relu1_2, relu2_2, relu3_3, relu4_3, relu5_2
"""


class AnimeGenerator(nn.Module):
    def __init__(self, input_nc=3, num_res=5):
        super().__init__()
        model1 = [nn.ReflectionPad2d(3),
                  nn.Conv2d(input_nc + 1, 32, 7),
                  nn.LeakyReLU(inplace=True)]

        model1 += [nn.ReflectionPad2d(3),
                   nn.Conv2d(32, 32, 7, stride=2),
                   nn.LeakyReLU(inplace=True)]

        for i in range(num_res):
            model1 += [ResBlock(32, 7)]

        model1 += [nn.ReflectionPad2d(1),
                   nn.Conv2d(32, 64, 3),
                   nn.LeakyReLU(inplace=True)]

        model2 = [nn.ReflectionPad2d(1),
                  nn.Conv2d(64, 32, 3),
                  nn.LeakyReLU(inplace=True),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(32, 32, 3, stride=2),
                  nn.LeakyReLU(inplace=True)]

        for i in range(num_res):
            model2 += [ResBlock(32)]

        model2 += [nn.ReflectionPad2d(1),
                   nn.Conv2d(32, 64, 3),
                   nn.LeakyReLU(inplace=True)]

        model7 = [nn.ReflectionPad2d(2),
                  nn.Conv2d(64, 32, 5),
                  nn.LeakyReLU(inplace=True),
                  nn.ReflectionPad2d(2),
                  nn.Conv2d(32, 32, 5, stride=2),
                  nn.LeakyReLU(inplace=True)]

        for i in range(num_res):
            model7 += [ResBlock(32, 5)]
        model7 += [nn.ReflectionPad2d(1),
                   nn.Conv2d(32, 64, 3),
                   nn.LeakyReLU(inplace=True)]

        model7_2 = [nn.ReflectionPad2d(2),
                    nn.Conv2d(64, 32, 5),
                    nn.LeakyReLU(inplace=True),
                    nn.ReflectionPad2d(2),
                    nn.Conv2d(32, 64, 5),
                    nn.LeakyReLU(inplace=True)]

        model8 = [nn.ReflectionPad2d(2),
                  nn.Conv2d(64, 32, 5),
                  nn.LeakyReLU(inplace=True),
                  nn.ReflectionPad2d(2),
                  nn.Conv2d(32, 32, 5, stride=2),
                  nn.LeakyReLU(inplace=True)]

        for i in range(num_res):
            model8 += [ResBlock(32, 5)]

        model8 += [nn.ReflectionPad2d(1),
                   nn.Conv2d(32, 64, 3),
                   nn.LeakyReLU(inplace=True)]

        model8_2 = [nn.ReflectionPad2d(2),
                    nn.Conv2d(64, 32, 5),
                    nn.LeakyReLU(inplace=True),
                    nn.ReflectionPad2d(2),
                    nn.Conv2d(32, 64, 5),
                    nn.LeakyReLU(inplace=True)]

        model3 = [nn.ReflectionPad2d(1),
                  nn.Conv2d(64, 32, 3),
                  nn.LeakyReLU(inplace=True),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(32, 3, 3),
                  nn.LeakyReLU(inplace=True)]

        model4 = [nn.ReflectionPad2d(1),
                  nn.Conv2d(64, 32, 3),
                  nn.LeakyReLU(inplace=True),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(32, 64, 3),
                  nn.LeakyReLU(inplace=True)]

        model5 = [nn.ReflectionPad2d(1),
                  nn.Conv2d(64, 16, 3),
                  nn.LeakyReLU(inplace=True),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(16, 3, 3),
                  nn.LeakyReLU(inplace=True)]

        model6 = [nn.ReflectionPad2d(3),
                  nn.Conv2d(3, 32, 7),
                  nn.LeakyReLU(inplace=True),
                  nn.ReflectionPad2d(3),
                  nn.Conv2d(32, 32, 7),
                  nn.LeakyReLU(inplace=True)
                  ]

        for i in range(5):
            model6 += [ResBlock(32, 7)]

        model6 += [nn.ReflectionPad2d(2),
                   nn.Conv2d(32, 32, 5),
                   nn.LeakyReLU(inplace=True),
                   nn.ReflectionPad2d(1),
                   nn.Conv2d(32, 3, 3),
                   nn.LeakyReLU(inplace=True)]
        """
        model9 = [nn.ReflectionPad2d(1),
                  nn.Conv2d(3, 32, 3),
                  nn.LeakyReLU(inplace=True),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(32, 64, 3),
                  nn.LeakyReLU(inplace=True)]
        """

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)
        self.model7_2 = nn.Sequential(*model7_2)
        self.model8 = nn.Sequential(*model8)
        self.model8_2 = nn.Sequential(*model8_2)
        # self.model9 = nn.Sequential(*model9)
        self.model3 = nn.Sequential(*model3)

    def forward(self, input_tensor, edges, device='cuda'):
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
        y = self.model6(new_tensor)
        new_b, new_g, new_r = torch.split(y, split_size_or_sections=1, dim=1)
        out_tensor.scatter_(1, index=max_indi, src=new_b)
        out_tensor.scatter_(1, index=med_indi, src=new_g)
        out_tensor.scatter_(1, index=min_indi, src=new_r)

        new_input = torch.cat((out_tensor, edges), dim=1)
        x1 = self.model1(new_input)
        x2 = self.model2(x1)
        x7 = self.model7(x2)
        x8 = self.model8(x7)
        x8 = interpolate(x8, (x7.shape[2], x7.shape[3]), mode='bilinear')
        cat_7_8 = x7 + x8
        x7 = self.model8_2(cat_7_8)
        x7 = interpolate(x7, (x2.shape[2], x2.shape[3]), mode='bilinear')
        cat_2_7 = x2 + x7
        x2 = self.model7_2(cat_2_7)
        x3 = interpolate(x2, size=(x1.shape[2], x1.shape[3]), mode='bilinear')
        cat_tensor = x1 + x3
        x4 = self.model4(cat_tensor)
        x4 = interpolate(x4, size=(input_tensor.shape[2], input_tensor.shape[3]), mode='bilinear')
        x5 = self.model3(x4)

        return x5


class ResBlock(nn.Module):
    def __init__(self, input_nc, k_size=3):
        super().__init__()
        padding_num = int(k_size / 2)
        model = [nn.ReflectionPad2d(padding_num),
                 nn.Conv2d(input_nc, input_nc, k_size),
                 nn.LeakyReLU(inplace=True),
                 nn.ReflectionPad2d(padding_num),
                 nn.Conv2d(input_nc, input_nc, k_size)]
        self.model = nn.Sequential(*model)

    def forward(self, input_tensor):
        return input_tensor + self.model(input_tensor)


class VGG16(nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = vgg16(pretrained=True).features
        conv1_2_list = []
        conv2_2_list = []
        conv3_3_list = []
        conv4_3_list = []
        for x in range(4):
            conv1_2_list.append(vgg_pretrained_features[x])
        self.conv1_2 = torch.nn.Sequential(*conv1_2_list)
        for x in range(4, 9):
            conv2_2_list.append(vgg_pretrained_features[x])
        self.conv2_2 = torch.nn.Sequential(*conv2_2_list)
        for x in range(9, 16):
            conv3_3_list.append(vgg_pretrained_features[x])
        self.conv3_3 = torch.nn.Sequential(*conv3_3_list)
        for x in range(16, 23):
            conv4_3_list.append(vgg_pretrained_features[x])
        self.conv4_3 = torch.nn.Sequential(*conv4_3_list)
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        relu1_2 = self.conv1_2(X)
        relu2_2 = self.conv2_2(relu1_2)
        relu3_3 = self.conv3_3(relu2_2)
        relu4_3 = self.conv4_3(relu3_3)
        return relu1_2, relu2_2, relu3_3, relu4_3


class TestGenerator(nn.Module):
    def __init__(self, input_nc=3, num_res=5):
        super().__init__()
        model1 = [nn.ReflectionPad2d(3),
                  nn.Conv2d(input_nc, 32, 7),
                  nn.LeakyReLU(inplace=True)]

        model1 += [nn.ReflectionPad2d(3),
                   nn.Conv2d(32, 32, 7, stride=2),
                   nn.LeakyReLU(inplace=True)]

        for i in range(num_res):
            model1 += [ResBlock(32, 7)]

        model1 += [nn.ReflectionPad2d(1),
                   nn.Conv2d(32, 64, 3),
                   nn.LeakyReLU(inplace=True)]

        model2 = [nn.ReflectionPad2d(1),
                  nn.Conv2d(64, 32, 3),
                  nn.LeakyReLU(inplace=True),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(32, 32, 3, stride=2),
                  nn.LeakyReLU(inplace=True)]

        for i in range(num_res):
            model2 += [ResBlock(32)]

        model2 += [nn.ReflectionPad2d(1),
                   nn.Conv2d(32, 64, 3),
                   nn.LeakyReLU(inplace=True)]

        model7 = [nn.ReflectionPad2d(2),
                  nn.Conv2d(64, 32, 5),
                  nn.LeakyReLU(inplace=True),
                  nn.ReflectionPad2d(2),
                  nn.Conv2d(32, 32, 5, stride=2),
                  nn.LeakyReLU(inplace=True)]

        for i in range(num_res):
            model7 += [ResBlock(32, 5)]
        model7 += [nn.ReflectionPad2d(1),
                   nn.Conv2d(32, 64, 3),
                   nn.LeakyReLU(inplace=True)]

        model7_2 = [nn.ReflectionPad2d(2),
                    nn.Conv2d(64, 32, 5),
                    nn.LeakyReLU(inplace=True),
                    nn.ReflectionPad2d(2),
                    nn.Conv2d(32, 64, 5),
                    nn.LeakyReLU(inplace=True)]

        model8 = [nn.ReflectionPad2d(2),
                  nn.Conv2d(64, 32, 5),
                  nn.LeakyReLU(inplace=True),
                  nn.ReflectionPad2d(2),
                  nn.Conv2d(32, 32, 5, stride=2),
                  nn.LeakyReLU(inplace=True)]

        for i in range(num_res):
            model8 += [ResBlock(32, 5)]

        model8 += [nn.ReflectionPad2d(1),
                   nn.Conv2d(32, 64, 3),
                   nn.LeakyReLU(inplace=True)]

        model8_2 = [nn.ReflectionPad2d(2),
                    nn.Conv2d(64, 32, 5),
                    nn.LeakyReLU(inplace=True),
                    nn.ReflectionPad2d(2),
                    nn.Conv2d(32, 64, 5),
                    nn.LeakyReLU(inplace=True)]

        model3 = [nn.ReflectionPad2d(1),
                  nn.Conv2d(64, 32, 3),
                  nn.LeakyReLU(inplace=True),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(32, 3, 3),
                  nn.LeakyReLU(inplace=True)]

        model4 = [nn.ReflectionPad2d(1),
                  nn.Conv2d(64, 32, 3),
                  nn.LeakyReLU(inplace=True),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(32, 64, 3),
                  nn.LeakyReLU(inplace=True)]

        model5 = [nn.ReflectionPad2d(1),
                  nn.Conv2d(64, 16, 3),
                  nn.LeakyReLU(inplace=True),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(16, 3, 3),
                  nn.LeakyReLU(inplace=True)]

        model6 = [nn.ReflectionPad2d(3),
                  nn.Conv2d(3, 32, 7),
                  nn.LeakyReLU(inplace=True),
                  nn.ReflectionPad2d(3),
                  nn.Conv2d(32, 32, 7),
                  nn.LeakyReLU(inplace=True)
                  ]

        for i in range(5):
            model6 += [ResBlock(32, 7)]

        model6 += [nn.ReflectionPad2d(2),
                   nn.Conv2d(32, 32, 5),
                   nn.LeakyReLU(inplace=True),
                   nn.ReflectionPad2d(1),
                   nn.Conv2d(32, 3, 3),
                   nn.LeakyReLU(inplace=True)]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)
        self.model7_2 = nn.Sequential(*model7_2)
        self.model8 = nn.Sequential(*model8)
        self.model8_2 = nn.Sequential(*model8_2)
        self.model3 = nn.Sequential(*model3)

    def forward(self, input_tensor):
        x1 = self.model1(input_tensor)
        x2 = self.model2(x1)
        x7 = self.model7(x2)
        x8 = self.model8(x7)
        x8 = interpolate(x8, (x7.shape[2], x7.shape[3]), mode='bilinear')
        cat_7_8 = x7 + x8
        x7 = self.model8_2(cat_7_8)
        x7 = interpolate(x7, (x2.shape[2], x2.shape[3]), mode='bilinear')
        cat_2_7 = x2 + x7
        x2 = self.model7_2(cat_2_7)
        x3 = interpolate(x2, size=(x1.shape[2], x1.shape[3]), mode='bilinear')
        cat_tensor = x1 + x3
        x4 = self.model4(cat_tensor)
        x4 = interpolate(x4, size=(input_tensor.shape[2], input_tensor.shape[3]), mode='bilinear')
        x5 = self.model3(x4)

        out_tensor = x5.detach().clone()
        max_tensor, max_indi = torch.max(x5, dim=1)
        median_tensor, median_indi = torch.median(x5, dim=1)
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
        y = self.model6(new_tensor)
        new_b, new_g, new_r = torch.split(y, split_size_or_sections=1, dim=1)
        out_tensor.scatter_(1, index=max_indi, src=new_b)
        out_tensor.scatter_(1, index=med_indi, src=new_g)
        out_tensor.scatter_(1, index=min_indi, src=new_r)

        return out_tensor

class TorchVGG16(nn.Module):
    def __init__(self):
        super().__init__()
        vgg_features = vgg19(True).features
        models = []
        for i in range(0, 26):
            models.append(vgg_features[i])
        self.model = nn.Sequential(*models)

    def forward(self, x):
        return self.model(x)


class Blur(nn.Module):
    def __init__(self, k_size=3):
        super().__init__()
        self.weight = torch.ones((3, 1, k_size, k_size), device='cuda') / (k_size * k_size)

    def forward(self, x):
        return conv2d(x, weight=self.weight, bias=None, groups=3, padding='same')


class VggModel16(nn.Module):
    def __init__(self):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg16(pretrained=True).features
        conv1_2_list = []
        conv2_2_list = []
        conv3_3_list = []
        conv4_3_list = []
        for x in range(4):
            conv1_2_list.append(vgg_pretrained_features[x])
        self.conv1_2 = torch.nn.Sequential(*conv1_2_list)
        for x in range(4, 9):
            conv2_2_list.append(vgg_pretrained_features[x])
        self.conv2_2 = torch.nn.Sequential(*conv2_2_list)
        for x in range(9, 16):
            conv3_3_list.append(vgg_pretrained_features[x])
        self.conv3_3 = torch.nn.Sequential(*conv3_3_list)
        for x in range(16, 23):
            conv4_3_list.append(vgg_pretrained_features[x])
        self.conv4_3 = torch.nn.Sequential(*conv4_3_list)

    def forward(self, X):
        relu1_2 = self.conv1_2(X)
        relu2_2 = self.conv2_2(relu1_2)
        relu3_3 = self.conv3_3(relu2_2)
        relu4_3 = self.conv4_3(relu3_3)
        return relu1_2, relu2_2, relu3_3, relu4_3



