from models import * 
import sys 
import os 


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
        f"./results/{args.model}/logdir/{args.mode}",
        f"./results/{args.model}/checkpoints",
        f"./results/{args.model}/logs"
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
