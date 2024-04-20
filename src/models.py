import os
import torch.nn as nn
import torch.nn.functional as  F
from transformers import ViTMAEModel


# Vit 模型
class VisionTransformer(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.mae = ViTMAEModel.from_pretrained(args.mae_path)
        self.classifier = nn.Linear(768, args.num_class)
        # for param in self.mae.parameters():
        #     param.requires_grad = False
    def forward(self, inputs):
        outputs = self.mae(inputs)
        logits = self.classifier(outputs[0][:, 0, :])
        
        return logits

 
# 卷积模型    
class ConvNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(2704, 512)
        self.fc2 = nn.Linear(512, 84)
        self.fc3 = nn.Linear(84, args.num_class)

    def forward(self, inputs):
        inputs = self.pool(F.relu(self.conv1(inputs)))
        inputs = self.pool(F.relu(self.conv2(inputs)))
        inputs = inputs.view(-1, self.num_flat_features(inputs))
        inputs = F.relu(self.fc1(inputs))
        inputs = F.relu(self.fc2(inputs))
        inputs = self.fc3(inputs)
        
        return inputs

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
    
# 配置模型
class Args:
    def __init__(self, root='./data', mode='train', log_path='') -> None:
        self.root = root
        self.mode = mode
        self.log_path = log_path if log_path else os.path.abspath('.') + '/ckpt/best.pth'
        self.restore = False  # 若为 True，表示恢复模型参数
        self.batch_size = 256
        self.image_size = 64
        self.epoch = 100
        self.num_class = 3755
        self.mae_path = '/home/karl/models/vit-mae-base'
        self.lr = 1e-3
        