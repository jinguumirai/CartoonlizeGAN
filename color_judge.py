import random
from utils import model_init, dis_init
import torch
import torch.nn as nn
from transformer_models import ColorDis, ColorDiscriminator, Dis
from torch.utils.data import Dataset, DataLoader
import glob
from PIL import Image
from models import Blur
from torchvision import transforms
import numpy as np


class ImageDataSet(Dataset):
    def __init__(self, image_path, transform_list):
        super().__init__()
        self.origin_image_names = glob.glob(image_path + '/ori_scenes/*.*')
        self.anime_image_names = glob.glob(r'Dataset/genshin_scenes/*.*')
        self.anime_face_names = glob.glob('Dataset/size256/*.*')
        self.real_face_names = glob.glob(r'Dataset/real_faces/*.*')
        self.origin_len = len(self.origin_image_names)
        self.transform_list = transform_list

    def __getitem__(self, item):
        ori_item = random.randint(0, self.origin_len - 1)
        origin_name = self.origin_image_names[ori_item]
        origin_img = Image.open(origin_name).convert('RGB')
        origin_img = self.transform_list(origin_img)
        anime_name = self.anime_image_names[item]
        anime_image = Image.open(anime_name).convert('RGB')
        anime_image = self.transform_list(anime_image)
        anime_face_item = random.randint(0, len(self.anime_face_names) - 1)
        real_face_item = random.randint(0, len(self.real_face_names) - 1)
        anime_face = Image.open(self.anime_face_names[anime_face_item]).convert('RGB')
        anime_face = self.transform_list(anime_face)
        real_face = Image.open(self.real_face_names[real_face_item]).convert('RGB')
        real_face = self.transform_list(real_face)

        return origin_img, anime_image, real_face, anime_face

    def __len__(self):
        return len(self.anime_image_names)


def random_simple():
    x = random.random()
    if x < 0.5:
        image_folders = glob.glob('Dataset/ori_scenes/*.*')
        target_label = 0.
    else:
        image_folders = glob.glob(r'C:\Users\jinguumirai\Pictures\genshin\*.*')
        target_label = 1.
    rand_index = random.randint(0, len(image_folders) - 1)
    image = Image.open(image_folders[rand_index]).convert('RGB')
    image = transforms.Resize(size=(320, 320))(image)
    image = transforms.ToTensor()(image)
    image = torch.unsqueeze(image, dim=0).cuda()
    image = blur_model(image)
    return image, target_label


discriminator = ColorDiscriminator().cuda()
discriminator.apply(dis_init)

blur_model = Blur(5).cuda()

dis_opti = torch.optim.Adam(discriminator.parameters(), lr=1e-3, betas=(0.9, 0.99), weight_decay=1e-4)

trans_list = transforms.Compose([transforms.Resize(size=(320, 320)),
                                 transforms.RandomHorizontalFlip(0.5),
                                 transforms.ToTensor()])
image_dataset = ImageDataSet('Dataset', trans_list)
image_loader = DataLoader(image_dataset, shuffle=True, batch_size=4)

dis_value = 0.

for k in range(1000):
    for i, image_tensor in enumerate(image_loader):
        if np.random.rand() > 0.5:
            origin_tensor = image_tensor[0].cuda()
            anime_tensor = image_tensor[1].cuda()
        else:
            origin_tensor = image_tensor[2].cuda()
            anime_tensor = image_tensor[3].cuda()
        blur_origin = blur_model(origin_tensor)
        blur_anime = blur_model(anime_tensor)
        fake_tensor = torch.zeros(size=(origin_tensor.shape[0],), dtype=torch.float).cuda()
        real_tensor = torch.ones(size=(origin_tensor.shape[0],), dtype=torch.float).cuda()

        fake_label = discriminator(blur_origin)
        fake_loss = nn.BCELoss()(fake_label, fake_tensor)

        real_label = discriminator(blur_anime)
        real_loss = nn.BCELoss()(real_label, real_tensor)

        dis_loss = real_loss + fake_loss
        dis_value = dis_loss.item()

        dis_opti.zero_grad()
        dis_loss.backward()
        dis_opti.step()

        if i % 10 == 0:
            print(dis_value)

    num = 0
    for j in range(100):
        test_tensor, test_label = random_simple()
        dis_test = discriminator(test_tensor)
        if dis_test < 0.5 and test_label == 0:
            num += 1
        elif dis_test > 0.5 and test_label == 1:
            num += 1

    print('accurate: %.4f' % (num / 100))
    torch.save(discriminator.state_dict(), 'dis.pth')