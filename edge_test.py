from models import ITVLoss
from PIL import Image
import torch
from torchvision import transforms

img_name = 'D:\\graduate\\03.jpg'
edge_name = 'D:\\graduate\\edge03.jpg'

img = Image.open(img_name).convert('RGB')
edge = Image.open(edge_name).convert('L')

img_tensor = transforms.ToTensor()(img)
img_tensor = torch.unsqueeze(img_tensor, dim=0).cuda()
edge_tensor = transforms.ToTensor()(edge)
edge_tensor = torch.unsqueeze(edge_tensor, dim=0).cuda()

itv = ITVLoss().cuda()
print(itv(img_tensor, edge_tensor))
