import torch
import torch.nn as nn
from transformer_models import GramDiscriminator, HSVGenerator, HSVGen
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import glob
from torchvision import transforms
import utils
from models import VggModel16, TVLoss
from torchvision.utils import save_image
import numpy as np
import random


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


generator = HSVGen().cuda()
generator.load_state_dict(torch.load('gram_out/20_gen.pth'))
trans_list = transforms.Compose([
                                 transforms.Resize(size=(299, 299)),
                                 transforms.ToTensor()])

image_dataset = ImageDataSet('Dataset', trans_list)
image_loader = DataLoader(image_dataset, shuffle=True, batch_size=1)
i = 0

for j, image_tensor in enumerate(image_loader):
    if np.random.rand() > 0.5:
        origin_tensor = image_tensor[0].cuda()
        anime_tensor = image_tensor[1].cuda()
    else:
        origin_tensor = image_tensor[2].cuda()
        anime_tensor = image_tensor[3].cuda()

    out_tensor = generator(origin_tensor)
    decompose_image = utils.decompose_tensor(out_tensor)
    save_image(decompose_image, 'Dataset/out_image/' + str(i) + '_origin.jpg')
    i += 1