from transformer_models import ColorDiscriminator
import torch
import glob
from PIL import Image
from torchvision import transforms
from models import Blur

dis = ColorDiscriminator().cuda()
dis.load_state_dict(torch.load('dis.pth'))
blur_model = Blur(5).cuda()
image_list = glob.glob(r'Dataset/genshin_scenes/*.*')

num = 0
all_num = 0

for name in image_list:
    all_num += 1
    img = Image.open(name).convert('RGB')
    img = transforms.Resize(size=(320, 320))(img)
    img = transforms.ToTensor()(img)
    img = torch.unsqueeze(img, dim=0).cuda()
    img = blur_model(img)
    dis_label = dis(img)
    if dis_label < 0.5:
        num += 1

print(num / all_num)

