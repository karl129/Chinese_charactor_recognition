import torch
import json
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as  F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, ViTMAEForPreTraining, ViTMAEModel
from PIL import Image
from .trainer import VisionTransformer

class Args_inference:
    def __init__(self):
        self.num_class = 3755
        self.mae_path = r'/Users/karl/work/day_work/汉字识别/ckpt/vit-mae-base'
        self.log_path = r'/Users/karl/work/day_work/汉字识别/ckpt/best.pth'
        self.res_dict_path = r'/Users/karl/work/day_work/汉字识别/src/char_dict.json'


def find_key_by_value(dictionary, value):
    for key, val in dictionary.items():
        if val == value:
            return key
    return None  # 如果未找到匹配的值，返回 None


def load_model():
    print('加载模型中....')
    args = Args_inference()
    model = VisionTransformer(args)
    model.eval()
    checkpoint = torch.load(args.log_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    print('模型加载完毕')
    return model

def inference(img_path, model):
    args = Args_inference()
    
    with open(args.res_dict_path, 'r') as f:
        r_dict = json.load(f)
    img = Image.open(img_path).convert('RGB')
    processor = AutoImageProcessor.from_pretrained(args.mae_path)
    image = torch.tensor(np.array(processor(img).pixel_values))

    output = model(image)
    _, pred = torch.max(output.data, 1)
    
    return find_key_by_value(r_dict, pred)


if __name__ == '__main__':
    pth = '../imgs/1866.png'
    print(inference(pth))