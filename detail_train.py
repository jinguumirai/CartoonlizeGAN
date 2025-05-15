import torch
import torch.nn as nn
from transformer_models import GenNet, ColorDiscriminator, HSVGenerator, ColorDis
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob
from torchvision import transforms
import utils
from models import Blur, TorchVGG19
from torchvision.utils import save_image


class ImageDataSet(Dataset):
    def __init__(self, image_path, transform_list):
        super().__init__()
        self.origin_image_names = glob.glob(image_path + '/ori_scenes/*.*')
        self.anime_image_names = glob.glob(image_path + '/genshin_scenes/*.*')
        self.transform_list = transform_list

    def __getitem__(self, item):
        origin_name = self.origin_image_names[item]
        origin_img = Image.open(origin_name).convert('RGB')
        origin_img = self.transform_list(origin_img)
        anime_name = self.anime_image_names[item]
        anime_image = Image.open(anime_name).convert('RGB')
        anime_image = self.transform_list(anime_image)
        return origin_img, anime_image

    def __len__(self):
        return len(self.anime_image_names)


generator = HSVGenerator().cuda()
generator.load_state_dict(torch.load('output_images/8_gen.pth'))
color_discriminator = ColorDiscriminator().cuda()
color_discriminator.load_state_dict(torch.load('output_images/8_color_dis.pth'))
vgg = TorchVGG19().cuda()
blur_model = Blur(9).cuda()

gen_opti = torch.optim.Adam(generator.parameters(), lr=9e-4, betas=(0.5, 0.9), eps=1e-5)
gen_opti.load_state_dict(torch.load('output_images/6_opti_gen.pth'))
dis_opti = torch.optim.Adam(color_discriminator.parameters(), lr=9e-4 * 1.025, betas=(0.5, 0.9), eps=1e-5)
dis_opti.load_state_dict(torch.load('output_images/6_dis_opti.pth'))

trans_list = transforms.Compose([transforms.RandomHorizontalFlip(0.5), transforms.ToTensor()])

image_dataset = ImageDataSet('Dataset', trans_list)
image_loader = DataLoader(image_dataset, shuffle=True, batch_size=4)

# generator.apply(utils.model_init)
# color_discriminator.apply(utils.model_init)

content_factor = 1e3 * 3.75
dis_factor = 1e3

for i in range(2000):
    out_image = None
    image_tensor = None
    origin_tensor = None
    dis_value = 0.
    content_value = 0.
    dis_real_value = 0.
    dis_value = 0.

    for j, image_tensor in enumerate(image_loader):
        origin_tensor = image_tensor[0].cuda()
        anime_tensor = image_tensor[1].cuda()
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

        out_tensor = generator(origin_tensor)
        compose_image = utils.compose_tensor(origin_tensor)

        vgg_origin = vgg(compose_image)
        vgg_out = vgg(out_tensor)

        content_loss = nn.MSELoss()(vgg_origin, vgg_out) * content_factor
        content_value = content_loss.item()

        decompose_image = utils.decompose_tensor(out_tensor)
        blur_out = blur_model(decompose_image)
        fake_label = color_discriminator(blur_out)
        dis_real_loss = nn.BCELoss()(fake_label, real_tensor) * dis_factor
        dis_real_value = dis_real_loss.item()

        gen_loss = content_loss + dis_real_loss
        gen_value = gen_loss.item()

        gen_opti.zero_grad()
        gen_loss.backward()
        gen_opti.step()

        if j % 10 == 0:
            print('epoch: %d, dis_loss: %.4f, content_loss: %.4f, dis_real_loss: %.4f, gen_loss: %.4f' % (
                i, dis_value, content_value, dis_real_value, gen_value))
            out_tensor = generator(origin_tensor)
            decompose_image = utils.decompose_tensor(out_tensor)

            torch.save(generator.state_dict(), 'more/' + str(i) + '_' + str(j) + '_gen.pth')
            torch.save(color_discriminator.state_dict(), 'more/' + str(i) + '_' + str(j) + '_color_dis.pth')
            torch.save(gen_opti.state_dict(), 'more/' + str(i) + '_' + str(j) + '_opti_gen.pth')
            torch.save(dis_opti.state_dict(), 'more/' + str(i) + '_' + str(j) + '_dis_opti.pth')
            save_image(origin_tensor, 'more/' + str(i) + '_' + str(j) + '_origin.jpg')
            save_image(decompose_image, 'more/' + str(i) + '_' + str(j) + '_out.jpg')







