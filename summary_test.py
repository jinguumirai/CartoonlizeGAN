import numpy as np
import time
from torchsummary import summary
from transformer_models import ColorDiscriminator,GramDiscriminator
from torchvision.models import vgg16, vgg19
import torch
from torch.nn.functional import interpolate


x = time.time()
model = vgg19(pretrained=True).cuda()
summary(model, (3, 320, 320))
print(time.time() - x)