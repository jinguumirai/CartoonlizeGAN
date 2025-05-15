from transformer_models import GenNet, NoHSVModel, TestGenerator, HSVGenerator
from PIL import Image
import glob
from torch.utils.data import Dataset, DataLoader
from os.path import basename
import torchvision
from torchvision import transforms
from models import TorchVGG19
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
vgg = TorchVGG19().cuda()
generator = HSVGenerator().cuda()

# generator.apply(utils.model_init)

opti = torch.optim.Adam(generator.parameters(), lr=1e-3, betas=(0.9, 0.99), eps=1e-5)

for i in range(2000):
    out_image = None
    image_tensor = None
    composed_smooth = None
    smooth_tensor = None
    smoothed_value = 0.
    for j, image_tensor in enumerate(image_loader):
        origin_tensor = image_tensor[0].cuda()
        smooth_tensor = image_tensor[1].cuda()
        out_image = generator(origin_tensor)
        composed_smooth = compose_tensor(smooth_tensor)
        vgg_smooth = vgg(composed_smooth)
        vgg_out = vgg(out_image)

        smoothed_loss = torch.nn.MSELoss()(vgg_smooth, vgg_out) * 1e5
        smoothed_value = smoothed_loss.item()

        opti.zero_grad()
        smoothed_loss.backward()
        opti.step()

        if j % 10 == 0:
            print('content loss: %.4f' % (smoothed_value, ))


    smoothed_tensor = decompose_tensor(composed_smooth)
    out_tensor = decompose_tensor(out_image)
    torchvision.utils.save_image(out_tensor, 'smooth_out.jpg')
    torchvision.utils.save_image(smoothed_tensor, 'smooth.jpg')
    torch.save(generator.state_dict(), 'smooth.pth')

