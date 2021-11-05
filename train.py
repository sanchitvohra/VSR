from dataset.dataloader import VSRDataset
from torch.utils.data import DataLoader
from models.vsr_model import VSRModel
from utils.config import load_config
import sys
import torch

def train(opt):
    model = VSRModel(opt)

    model.config_training()

    data = {}
    data['gt'] = torch.rand((1, 30, 3, 128 * 4, 128 * 4)).to(torch.float32).cuda()
    data['lr'] = torch.rand((1, 30, 3, 128, 128)).to(torch.float32).cuda()
    model.train_batch(data)
    exit()

opt = load_config(sys.argv[1])

train(opt)
