import os
import argparse
import logging
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .data_handler import classes_txt
from .my_dataset import MyDataset
from .models import VisionTransformer, ConvNet, Args


def train(args):
    logging.basicConfig(filename='train.log',                             # 指定日志文件名
                    filemode='w',                                         # 指定文件打开模式
                    level=logging.INFO,                                   # 指定日志级别，低于此级别的日志将被忽略
                    format='{asctime} - {lineno} - {levelname}:{message}',  # 指定日志格式 
                    style='{',                                            # 指定格式字符串的风格
                    datefmt='%Y-%m-%d %H:%M:%S')                          # 指定日期格式
    
    train_set = MyDataset(args.root + '/train.txt', num_class=args.num_class, transforms=None, args=args)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Start training at device {device}')

    model = VisionTransformer(args)
    
    model.to(device)

    model.train()

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    
    param_groups = [{'params': model.mae.parameters(), 'lr': args.lr * 0.01},
                    {'params': model.classifier.parameters(), 'lr': args.lr}]
    optimizer = optim.Adam(param_groups, lr=args.lr)
    
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

            if global_step % 10 == 0:  # every 10 steps
                logging.info('epoch %5d: batch: %5d, loss: %f' % (epoch + 1, global_step + 1, running_loss / 10))
                running_loss = 0.0
                
        scheduler.step(loss) 
       
        acc = validation(args, model)
        model.train()
        if acc > best_acc:
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model is None:
        model = VisionTransformer(args)
        model.to(device)
        checkpoint = torch.load(args.log_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    total = 0.0
    correct = 0.0
    with torch.no_grad():
        for data in tqdm(test_loader):
            inputs, labels = data[0][0].to(device), data[1]
            outputs = model(inputs)
            _, predict = torch.max(outputs.data, 1)
            predict = predict.cpu()
            total += labels.size(0)

            correct += (predict == labels).sum().item()
            
    acc = (correct / total * 100)
    logging.info('Accuracy: %.2f%%' % (correct / total * 100))

    return acc


if __name__ == '__main__':
    args = Args()
    classes_txt(args.root + '/train', args.root + '/train.txt', num_class=args.num_class)
    classes_txt(args.root + '/test', args.root + '/test.txt', num_class=args.num_class)
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'validation':
        validation(args)