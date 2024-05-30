from models import * 
import sys 
import os 

# 解析 images.txt 文件来获取所有图片的路径
def parse_images_file(file_path):
    image_paths = {}
    with open(file_path, 'r') as f:
        for line in f.readlines():
            img_id, path = line.strip().split(' ')
            image_paths[int(img_id)] = path
    return image_paths

# 解析 image_class_labels.txt 文件来获取每张图片对应的类别
def parse_labels_file(file_path):
    image_labels = {}
    with open(file_path, 'r') as f:
        for line in f.readlines():
            img_id, label = line.strip().split(' ')
            image_labels[int(img_id)] = int(label) - 1  # 类别从0开始计数
    return image_labels

# 解析 train_test_split.txt 文件来获取每张图片是属于训练集还是测试集
def parse_split_file(file_path):
    image_splits = {}
    with open(file_path, 'r') as f:
        for line in f.readlines():
            img_id, is_train = line.strip().split(' ')
            image_splits[int(img_id)] = int(is_train)
    return image_splits

def get_network(name):
    if name == "vgg16":
        net = vgg16_bn()
    elif name == "vgg13":
        net = vgg13_bn()
    elif name == 'vgg11':
        net = vgg11_bn()
    elif name == 'vgg19':
        net = vgg19_bn()
    elif name == 'resnet18':
        net = ResNet18()
    elif name == 'resnet34':
        net = ResNet34()
    elif name == 'resnet50':
        net = ResNet50()
    elif name == 'resnet101':
        net = ResNet101()
    elif name == 'resnet152':
        net = ResNet152()
    elif name == "googlenet":
        net = GoogleNet()
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()
    
    return net 

def make_folders(args):
    folders = [
        f"./results/{args.model}",
        f"./results/{args.model}/logdir",
        f"./results/{args.model}/checkpoints",
        f"./results/{args.model}/logs"
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
# /{args.mode}