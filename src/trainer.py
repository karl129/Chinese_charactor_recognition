import os
import argparse
import logging
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as  F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, ViTMAEForPreTraining, ViTMAEModel
from PIL import Image


logging.basicConfig(filename='train.log',                               # 指定日志文件名
                    filemode='w',                                         # 指定文件打开模式
                    level=logging.INFO,                                   # 指定日志级别，低于此级别的日志将被忽略
                    format='{asctime} - {lineno} - {levelname}:{message}',  # 指定日志格式 
                    style='{',                                            # 指定格式字符串的风格
                    datefmt='%Y-%m-%d %H:%M:%S')                          # 指定日期格式
    

class MyDataset(Dataset):
    def __init__(self, txt_path, num_class, args,transforms=None):
        super(MyDataset, self).__init__()
        self.processor = AutoImageProcessor.from_pretrained(args.mae_path)
        
        images = []
        labels = []
        with open(txt_path, 'r') as f:
            for line in f:
                if int(line.split('/')[-2]) >= num_class:  # just get images of the first #num_class
                    break
                line = line.strip('\n')
                images.append(line)
                labels.append(int(line.split('/')[-2]))
        self.images = images
        self.labels = labels
        self.transforms = transforms

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        label = self.labels[index]
        image = self.processor(img).pixel_values
        if self.transforms is not None:
            image = self.transforms(image)
        return image, label

    def __len__(self):
        return len(self.labels)

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
    
class NetSmall(nn.Module):
    def __init__(self):
        super(NetSmall, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(2704, 512)
        self.fc2 = nn.Linear(512, 84)
        self.fc3 = nn.Linear(84, args.num_class)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def train(args):
    train_set = MyDataset(args.root + '/train.txt', num_class=args.num_class, transforms=None, args=args)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Start training at device {device}')

    model = VisionTransformer(args)
    
    model.to(device)

    model.train()

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    
    param_groups = [{'params': model.mae.parameters(), 'lr': args.lr * 0.01},
                    {'params': model.classifier.parameters(), 'lr': args.lr}]
    optimizer = optim.Adam(param_groups, lr=args.lr)
    
    # 学习率调整函数
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    
    if args.restore:
        checkpoint = torch.load(args.log_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']
        epoch = checkpoint['epoch']
    else:
        loss = 0.0
        epoch = 0
    best_acc = 0.0
    while epoch < args.epoch:
        logging.info(f'{epoch+1}/{args.epoch}')
        running_loss = 0.0
        global_step = 0
        for data in tqdm(train_loader):
            global_step += 1
            inputs, labels = data[0][0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outs = model(inputs)
            loss = criterion(outs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if global_step % 10 == 0:  # every 200 steps
                # print('epoch %5d: batch: %5d, loss: %f' % (epoch + 1, global_step + 1, running_loss / 10))
                logging.info('epoch %5d: batch: %5d, loss: %f' % (epoch + 1, global_step + 1, running_loss / 10))
                running_loss = 0.0
                
        scheduler.step(loss) 
       
        acc = validation(args, model)
        model.train()
        if acc > best_acc:
            # print('Save checkpoint...')
            logging.info(f'Saving checkpoint after {epoch} with ACC: {acc}')
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss},
                        args.log_path)
        
        epoch += 1

    print('Finish training')


def validation(args, model=None):

    test_set = MyDataset(args.root + '/test.txt', num_class=args.num_class, transforms=None, args=args)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    
    if model is None:
        # model = NetSmall()
        model = VisionTransformer(args)
        model.to(device)
        checkpoint = torch.load(args.log_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    total = 0.0
    correct = 0.0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels = data[0][0].to(device), data[1]
            outputs = model(inputs)
            _, predict = torch.max(outputs.data, 1)
            predict = predict.cpu()
            total += labels.size(0)

            correct += (predict == labels).sum().item()

            # if i % 100 == 99:
            #     print('batch: %5d,\t acc: %f' % (i + 1, correct / total))
    acc = (correct / total * 100)
    # print('Accuracy: %.2f%%' % (correct / total * 100))
    logging.info('Accuracy: %.2f%%' % (correct / total * 100))

    return acc


# 配置模型
class Args:
    def __init__(self, root='./data', mode='train', log_path='') -> None:
        self.root = root
        self.mode = mode
        self.log_path = log_path if log_path else os.path.abspath('.') + '/ckpt/best.pth'
        self.restore = False  # 默认为 True，表示恢复检查点
        self.batch_size = 256
        self.image_size = 64
        self.epoch = 100
        self.num_class = 3755  # 默认为 100，可以根据需要修改范围
        self.mae_path = '/home/karl/models/vit-mae-base'
        self.lr = 1e-3

if __name__ == '__main__':

    args = Args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'validation':
        validation(args)