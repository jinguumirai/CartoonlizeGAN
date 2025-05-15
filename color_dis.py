import torch
import torch.nn as nn
from transformer_models import GenNet, ColorDiscriminator, HSVGenerator, ColorDis, HSVGen
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob
from torchvision import transforms
import utils
from models import Blur, TorchVGG16, TVLoss
from torchvision.utils import save_image
import random
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


generator = HSVGen().cuda()
color_discriminator = ColorDiscriminator().cuda()
# vgg = TorchVGG16().cuda()
blur_model = Blur(7).cuda()
tv_model = TVLoss().cuda()

gen_opti = torch.optim.Adam(generator.parameters(), lr=1e-3, betas=(0.85, 0.99), eps=1e-8)
sche_gen = torch.optim.lr_scheduler.LambdaLR(gen_opti, utils.lambda_lr)
dis_opti = torch.optim.Adam(color_discriminator.parameters(), lr=2e-3, betas=(0.85, 0.99), eps=1e-8)
sche_dis = torch.optim.lr_scheduler.LambdaLR(dis_opti, utils.lambda_lr)

trans_list = transforms.Compose([transforms.Resize(size=(320, 320)),
                                 transforms.RandomHorizontalFlip(0.5),
                                 transforms.ToTensor()])

image_dataset = ImageDataSet('Dataset', trans_list)
image_loader = DataLoader(image_dataset, shuffle=True, batch_size=4)

generator.load_state_dict(torch.load('vgg_model/pre_model5.pth'))
# generator.apply(utils.model_init)
color_discriminator.apply(utils.model_init)

content_factor = 1
dis_factor = 1
fight_factor = 0.35
tv_factor = 4e5

for i in range(2000):
    out_image = None
    image_tensor = None
    origin_tensor = None
    dis_value = 0.
    content_value = 0.
    dis_real_value = 0.
    dis_value = 0.
    tv_item = 0

    for j, image_tensor in enumerate(image_loader):
        if np.random.rand() > 0.5:
            origin_tensor = image_tensor[0].cuda()
            anime_tensor = image_tensor[1].cuda()
        else:
            origin_tensor = image_tensor[2].cuda()
            anime_tensor = image_tensor[3].cuda()
        fake_tensor = torch.zeros(size=(origin_tensor.shape[0], ), dtype=torch.float).cuda()
        real_tensor = torch.ones(size=(origin_tensor.shape[0], ), dtype=torch.float).cuda()
        out_tensor = generator(origin_tensor)
        decompose_image = utils.decompose_tensor(out_tensor)
        blur_out = blur_model(decompose_image)

        fake_label = color_discriminator(blur_out).view(-1)
        fake_loss = nn.BCELoss()(fake_label, fake_tensor)

        blur_anime = blur_model(anime_tensor)

        real_label = color_discriminator(blur_anime).view(-1)
        real_loss = nn.BCELoss()(real_label, real_tensor)

        dis_loss = (real_loss + fake_loss) * dis_factor
        dis_value = dis_loss.item()

        dis_opti.zero_grad()
        dis_loss.backward()
        dis_opti.step()
        sche_dis.step()

        out_tensor = generator(origin_tensor)
        compose_image = utils.compose_tensor(origin_tensor)

        content_loss = nn.MSELoss()(compose_image, out_tensor) * content_factor
        content_value = content_loss.item()

        decompose_image = utils.decompose_tensor(out_tensor)
        blur_out = blur_model(decompose_image)
        fake_label = color_discriminator(blur_out)
        dis_real_loss = nn.BCELoss()(fake_label, real_tensor) * fight_factor
        dis_real_value = dis_real_loss.item()

        tv_loss = tv_model(out_tensor) * tv_factor
        tv_item = tv_loss.item()

        gen_loss = content_loss + dis_real_loss + tv_loss
        gen_value = gen_loss.item()

        gen_opti.zero_grad()
        gen_loss.backward()
        gen_opti.step()
        sche_gen.step()

        if j % 10 == 0:
            print('epoch: %d, dis_loss: %.4f, content_loss: %.4f, dis_real_loss: %.4f, tv_loss: %.4f, gen_loss: %.4f' % (
                i, dis_value, content_value, dis_real_value, tv_item, gen_value))

    out_tensor = generator(origin_tensor)
    decompose_image = utils.decompose_tensor(out_tensor)

    torch.save(generator.state_dict(), 'output_images/' + str(i) + '_gen.pth')
    torch.save(color_discriminator.state_dict(), 'output_images/' + str(i) + '_color_dis.pth')
    torch.save(gen_opti.state_dict(), 'output_images/' + str(i) + '_opti_gen.pth')
    torch.save(dis_opti.state_dict(), 'output_images/' + str(i) + '_dis_opti.pth')
    torch.save(sche_dis.state_dict(), 'output_images/' + str(i) + '_sche_dis.pth')
    torch.save(sche_gen.state_dict(), 'output_images/' + str(i) + '_sche_gen.pth')
    save_image(origin_tensor, 'output_images/' + str(i) + '_origin.jpg')
    save_image(decompose_image, 'output_images/' + str(i) + '_out.jpg')




