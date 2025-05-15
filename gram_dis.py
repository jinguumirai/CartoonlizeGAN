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
gram_dis= GramDiscriminator().cuda()
vgg = VggModel16().cuda()
tv_model = TVLoss().cuda()

gen_opti = torch.optim.Adam(generator.parameters(), lr=1e-3, betas=(0.4, 0.99))
sche_gen = torch.optim.lr_scheduler.LambdaLR(gen_opti, utils.lambda_lr)
dis_opti = torch.optim.Adam(gram_dis.parameters(), lr=1e-3, betas=(0.4, 0.99), weight_decay=1e-3)
sche_dis = torch.optim.lr_scheduler.LambdaLR(dis_opti, utils.lambda_lr)

trans_list = transforms.Compose([transforms.RandomHorizontalFlip(0.5),
                                 transforms.Resize(size=(320, 320)),
                                 transforms.ToTensor()])

image_dataset = ImageDataSet('Dataset', trans_list)
image_loader = DataLoader(image_dataset, shuffle=True, batch_size=3)

generator.load_state_dict(torch.load('pre_model/pre_model5.pth'))
# generator.apply(utils.model_init)
gram_dis.apply(utils.dis_init)

content_factor = 1
dis_factor = 1
compete_factor = 0.185
tv_factor = 2e5

for i in range(2000):
    out_tensor = None
    image_tensor = None
    origin_tensor = None
    vgg_out = None
    real_tensor = None
    fake_tensor = None
    gram_out = None
    vgg_anime = None
    gram_anime = None
    dis_loss = None
    gen_loss = None
    real_label = None
    dis_value = 0.
    content_value = 0.
    dis_real_value = 0.
    dis_value = 0.
    tv_item = 0.


    for j, image_tensor in enumerate(image_loader):
        if np.random.rand() > 0.5:
            origin_tensor = image_tensor[0].cuda()
            anime_tensor = image_tensor[1].cuda()
        else:
            origin_tensor = image_tensor[2].cuda()
            anime_tensor = image_tensor[3].cuda()
        fake_tensor = torch.zeros(size=(origin_tensor.shape[0],), dtype=torch.float).cuda()
        real_tensor = torch.ones(size=(origin_tensor.shape[0],), dtype=torch.float).cuda()
        out_tensor = generator(origin_tensor)
        vgg_out = vgg(out_tensor)
        gram_out = utils.gramian_mat(vgg_out)

        fake_label =gram_dis(gram_out).view(-1)
        fake_loss = nn.BCELoss()(fake_label, fake_tensor)

        compose_anime = utils.compose_tensor(anime_tensor)
        vgg_anime = vgg(compose_anime)
        gram_anime = utils.gramian_mat(vgg_anime)

        real_label = gram_dis(gram_anime).view(-1)
        real_loss = nn.BCELoss()(real_label, real_tensor)

        dis_loss = (real_loss + fake_loss) * dis_factor
        dis_value = dis_loss.item()

        dis_opti.zero_grad()
        dis_loss.backward()
        dis_opti.step()
        sche_dis.step()

        out_tensor = generator(origin_tensor)
        compose_image = utils.compose_tensor(origin_tensor)

        vgg_origin = vgg(compose_image)
        vgg_out = vgg(out_tensor)

        content_loss = nn.MSELoss()(compose_image, out_tensor) * content_factor
        content_value = content_loss.item()

        gram_out = utils.gramian_mat(vgg_out)
        fake_label = gram_dis(gram_out)
        dis_real_loss = nn.BCELoss()(fake_label, real_tensor) * compete_factor
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
            print('epoch: %d, dis_loss: %.4f, content_loss: %.4f, dis_real_loss: %.4f, tv_loss: %.6f, gen_loss: %.4f' % (
                i, dis_value, content_value, dis_real_value, tv_item, gen_value))

    out_tensor = generator(origin_tensor)
    decompose_image = utils.decompose_tensor(out_tensor)

    torch.save(generator.state_dict(), 'gram_out/' + str(i) + '_gen.pth')
    torch.save(gram_dis.state_dict(), 'gram_out/' + str(i) + '_gram_dis.pth')
    torch.save(gen_opti.state_dict(), 'gram_out/' + str(i) + '_opti_gen.pth')
    torch.save(dis_opti.state_dict(), 'gram_out/' + str(i) + '_dis_opti.pth')
    torch.save(sche_dis.state_dict(), 'gram_out/' + str(i) + '_sche_dis.pth')
    torch.save(sche_gen.state_dict(), 'gram_out/' + str(i) + '_sche_gen.pth')
    save_image(origin_tensor, 'gram_out/' + str(i) + '_origin.jpg')
    save_image(decompose_image, 'gram_out/' + str(i) + '_out.jpg')




