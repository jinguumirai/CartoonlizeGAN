from transformer_models import GenNet, NoHSVModel, HSVGenerator, HSVGen, TransformerNet
from PIL import Image
import glob
from torch.utils.data import Dataset, DataLoader
from os.path import basename
import torchvision
from torchvision import transforms
from models import TorchVGG16
import utils
import torch
from utils import decompose_tensor, compose_tensor


class SmoothImageDataSet(Dataset):
    def __init__(self, image_path, transform_list):
        super().__init__()
        self.origin_image_names = glob.glob(image_path + '/ori_scenes/*.*')
        self.smoothed_folders = image_path + '/smoothed_images/'
        self.transform_list = transform_list

    def __getitem__(self, item):
        origin_name = self.origin_image_names[item]
        origin_img = Image.open(origin_name).convert('RGB')
        origin_img = self.transform_list(origin_img)
        smoothed_image = Image.open(self.smoothed_folders + basename(origin_name)).convert('RGB')
        smoothed_image = self.transform_list(smoothed_image)
        return origin_img, smoothed_image

    def __len__(self):
        return len(self.origin_image_names)


transform_list = transforms.Compose([transforms.ToTensor()])

image_dataset = SmoothImageDataSet('Dataset', transform_list)
image_loader = DataLoader(image_dataset, shuffle=True, batch_size=4)
vgg = TorchVGG16().cuda()
generator = HSVGen().cuda()

generator.apply(utils.model_init)

opti = torch.optim.Adam(generator.parameters(), lr=4e-4, betas=(0.8, 0.99))

for i in range(2000):
    out_image = None
    origin_tensor = None
    content_value = 0.
    composed = None
    for j, image_tensor in enumerate(image_loader):
        origin_tensor = image_tensor[0].cuda()
        composed = utils.compose_tensor(origin_tensor)
        out_image = generator(origin_tensor)
        vgg_out = vgg(out_image)
        vgg_composed = vgg(composed)

        content_loss = torch.nn.MSELoss()(out_image, composed)
        content_value = content_loss.item()

        opti.zero_grad()
        content_loss.backward()
        opti.step()

        if j % 10 == 0:
            print('content loss: %.4f' % (content_value, ))

    out_tensor = decompose_tensor(out_image)
    torchvision.utils.save_image(out_tensor, 'vgg_model/content' + str(i) + '.jpg')
    torchvision.utils.save_image(origin_tensor, 'vgg_model/origin' + str(i) + '.jpg')
    torch.save(generator.state_dict(), 'vgg_model/pre_model' + str(i) + '.pth')

