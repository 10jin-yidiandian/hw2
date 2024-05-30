import os
from torchvision import transforms
import torch
from torchvision.models import resnet18
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn as nn
from tensorboardX import SummaryWriter
from utils import make_folders, parse_images_file, parse_labels_file,parse_split_file
from models import ResNet18
from dataset import CUBDataset, data_transforms
import time 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="resnet18")
parser.add_argument("--epoch", type=int, default=200)
parser.add_argument("--batchsize", type=int, default=128)
parser.add_argument("--pretrained", type=bool, default=True)
args = parser.parse_args()

# 解析文件
image_paths = parse_images_file('CUB_200_2011/images.txt')
image_labels = parse_labels_file('CUB_200_2011/image_class_labels.txt')
image_splits = parse_split_file('CUB_200_2011/train_test_split.txt')

# 创建数据集
train_dataset = CUBDataset('CUB_200_2011', image_paths, image_labels, image_splits, train=True, transform=data_transforms['train'])
val_dataset = CUBDataset('CUB_200_2011', image_paths, image_labels, image_splits, train=False, transform=data_transforms['val'])

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batchsize, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.pretrained:
    model = resnet18(pretrained=True)
    Pretrained = 'Pretrained'
else:
    model = resnet18(pretrained=False)
    Pretrained = 'Untrained'
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 200)
model = model.to(device)

make_folders(args)
learning_rate = 0.01
writer = SummaryWriter(f"./results/{args.model}/logdir/{Pretrained}_{learning_rate}_{args.epoch}")

# 获取模型所有参数
params_to_update = []
params_names = []

# 遍历模型的所有参数，只更新 requires_grad 为 True 的参数
for name, param in model.named_parameters():
    if param.requires_grad:
        if name not in params_names:  # 确保参数不重复添加
            params_to_update.append(param)
            params_names.append(name)

# 定义优化器
optimizer = optim.SGD(params_to_update, lr=learning_rate, momentum=0.9)#, weight_decay = 5e-4
train_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.2) 
# 训练和验证模型
num_epochs = args.epoch
best_acc = 0
start_time = time.time()
for epoch in range(num_epochs):
    print(f'Epoch {epoch}/{num_epochs - 1}')
    print('-' * 10)

    # 每个epoch都有训练和验证阶段
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # 训练模式
            dataloader = train_loader
        else:
            model.eval()  # 验证模式
            dataloader = val_loader

        running_loss = 0.0
        running_corrects = 0

        # 迭代数据
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 每次迭代都要清空梯度
            optimizer.zero_grad()

            # 前向传播
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = torch.nn.CrossEntropyLoss()(outputs, labels)

                # 训练时才进行反向传播和优化
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = running_corrects.double() / len(dataloader.dataset)

        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        writer.add_scalar(f"{phase}_loss", epoch_loss, epoch)
        writer.add_scalar(f"{phase}_acc", epoch_acc, epoch)
        if epoch_acc > best_acc and phase=='val':
            best_acc = epoch_acc
        #     torch.save(model, f"results/{args.model}/checkpoints/cifar_{args.pretrained}_{args.epoch}_{learning_rate}.pth")
        # train_scheduler.step()
    print()

writer.close()
end_time = time.time()
print(end_time-start_time)
torch.save(model, f"./results/{args.model}/checkpoints/CUB_{Pretrained}_{args.epoch}_{learning_rate}.pth")#
with open(f"./results/{args.model}/logs/{Pretrained}_{args.epoch}_{learning_rate}.log", "w+") as f:
    f.write(f"Epoch:{args.epoch}, lr:{learning_rate}, Name:{args.model}, Best Accuracy:{best_acc}, Last Accuracy:{epoch_acc}")
print('Training complete')