import torch
from models import AnimeGenerator
import numpy as np

generator = AnimeGenerator(3, num_res=5).cuda()
opti = torch.optim.Adam(lr=1e-3, params=generator.parameters())
