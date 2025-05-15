import torchvision.utils
import torch.nn as nn
import utils
from transformer_models import TestGenerator, GenNet
from models import TorchVGG16, TorchVGG19
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from utils import TestImageDataSet, decompose_tensor, compose_tensor


transform_list = transforms.Compose([transforms.ToTensor()])

image_dataset = TestImageDataSet('Dataset/ori_scenes', transform_list)
image_loader = DataLoader(image_dataset, shuffle=True, batch_size=4)
vgg = TorchVGG19().cuda()
generator = GenNet().cuda()

generator.apply(utils.model_init)

opti = torch.optim.Adam(generator.parameters(), lr=5e-4)

instance_norm = nn.InstanceNorm2d(3).cuda()

for i in range(2000):
    out_image = None
    image_tensor = None
    composed_input = None
    content_value = 0.
    for j, image_tensor in enumerate(image_loader):
        image_tensor = image_tensor.cuda()
        out_image = generator(image_tensor)
        composed_input = compose_tensor(image_tensor)
        vgg_in = vgg(composed_input)
        vgg_out = vgg(out_image)

        content_loss = torch.nn.MSELoss()(vgg_in, vgg_out) * 1e5
        content_value = content_loss.item()

        opti.zero_grad()
        content_loss.backward()
        opti.step()

        if j % 10 == 0:
            print('content loss: %.4f' % (content_value, ))


    decompose_images = decompose_tensor(out_image)
    origin_tensor = decompose_tensor(composed_input)
    torchvision.utils.save_image(origin_tensor, 'origin.jpg')
    torchvision.utils.save_image(decompose_images, 'test.jpg')

