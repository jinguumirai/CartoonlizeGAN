import glob
from torch.utils.data import Dataset
from PIL import Image
import torch
import numpy as np


class TestImageDataSet(Dataset):
    def __init__(self, image_path, transform_list):
        super().__init__()
        self.image_names = glob.glob(image_path + '/*.*')
        self.transform_list = transform_list

    def __getitem__(self, item):
        image = Image.open(self.image_names[item]).convert('RGB')
        image = self.transform_list(image)
        return image

    def __len__(self):
        return len(self.image_names)


def decompose_tensor(composed_tensor):
    r, g, b = torch.split(composed_tensor, 1, dim=1)
    mean_list = np.array([0.485, 0.456, 0.406])
    std_list = np.array([0.229, 0.224, 0.225])
    new_r = r * std_list[0] + mean_list[0]
    new_g = g * std_list[1] + mean_list[1]
    new_b = b * std_list[2] + mean_list[2]
    decomposed_img = torch.cat((new_r, new_g, new_b), dim=1)
    return decomposed_img

def compose_tensor(input_tensor):
    r, g, b = torch.split(input_tensor, 1, dim=1)
    mean_list = np.array([0.485, 0.456, 0.406])
    std_list = np.array([0.229, 0.224, 0.225])
    new_r = (r - mean_list[0]) / std_list[0]
    new_g = (g - mean_list[1]) / std_list[1]
    new_b = (b - mean_list[2]) / std_list[2]
    return torch.cat((new_r, new_g, new_b), dim=1)


def model_init(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 1e-3)
        torch.nn.init.normal_(m.bias.data, 1e-6)
    elif class_name.find('Linear') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 1e-3)
        torch.nn.init.normal_(m.bias.data, 1e-6)


def dis_init(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0., 1e-1)
        torch.nn.init.normal_(m.bias.data, 0., 1e-4)
    elif class_name.find('Linear') != -1:
        torch.nn.init.normal_(m.weight.data, 0., 1e-1)
        torch.nn.init.normal_(m.bias.data, 0., 1e-4)


def lambda_lr(epoch):
        return max(1 - epoch * 6e-6, 0.1)

def gramian_mat(feature1):
    def gram(fea):
        b, c, h, w = fea.shape
        fea = fea.view(b, c, h * w)
        trans_fea = fea.transpose(1, 2)
        gram_result = fea.bmm(trans_fea) / (c * h * w)
        return gram_result
    return_list = []
    for fea1 in feature1:
        gram1 = gram(fea1)
        gram1 = torch.unsqueeze(gram1, dim=1)
        return_list.append(gram1)
    return return_list