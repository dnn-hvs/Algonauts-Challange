import numpy as np

import scipy.io as sio
import glob
import zipfile
import numpy as np
import torch
from vgg import *
from resnet import *
from alexnet import *
from squeezenet import *
from densenet import *
from inceptionv3 import *
from googlenet import *

from torch.autograd import Variable as V
from tqdm import tqdm
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image

models = {
    'alex1': AlexNet(),
    'alex2': AlexNet(),
    'vgg1': vgg16(),
    'vgg2': vgg16(),
    'alexnet': AlexNet(),
    'vgg11': vgg11(pretrained=True),
    'vgg13': vgg13(pretrained=True),
    'vgg16': vgg16(pretrained=True),
    'vgg19': vgg19(pretrained=True),
    'sqnet1_0': SqueezeNet1_0(),
    'sqnet1_1': SqueezeNet1_1(),
    'resnet18': resnet18(pretrained=True),
    'resnet34': resnet34(pretrained=True),
    'resnet50': resnet50(pretrained=True),
    'resnet101': resnet101(pretrained=True),
    'resnet152': resnet152(pretrained=True),
    'densenet121': densenet121(pretrained=True),
    'densenet161': densenet161(pretrained=True),
    'densenet169': densenet169(pretrained=True),
    'densenet201': densenet201(pretrained=True),
    'googlenet': googlenet(pretrained=True),
    'inception': inception_v3(pretrained=True)
}


def zip(src, dst):
    zf = zipfile.ZipFile("%s.zip" % (dst), "w", zipfile.ZIP_DEFLATED)
    abs_src = os.path.abspath(src)
    for dirname, subdirs, files in os.walk(src):
        for filename in files:
            absname = os.path.abspath(os.path.join(dirname, filename))
            arcname = absname[len(abs_src) + 1:]
            print("zipping {} as {}".format(
                os.path.join(dirname, filename), arcname))
            zf.write(absname, arcname)
    zf.close()


def execute_model(image_dir, net_save_dir, model):
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    centre_crop = trn.Compose([
        trn.Resize((224, 224)),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_list = glob.glob(image_dir + "/*.jpg")
    image_list.sort()
    image_list = image_list

    for image in image_list:
        img = Image.open(image)
        filename = image.split("/")[-1].split(".")[0]
        input_img = V(centre_crop(img).unsqueeze(0))
        print(input_img.size(), filename)
        print(input_img.size())
        if torch.cuda.is_available():
            input_img = input_img.cuda()
        x = model.forward(input_img)
        save_path = os.path.join(net_save_dir, filename+".mat")
        feats = {}
        for i, feat in tqdm(enumerate(x)):
            print(save_path)
            print(feat.data.cpu().numpy().shape)
            feats[model.feat_list[i]] = feat.data.cpu().numpy()
        sio.savemat(save_path, feats)
        print(str(feats.keys()))


def load_model(model, model_path, optimizer=None, resume=False,
               lr=None):
    start_epoch = 0
    checkpoint = torch.load(
        model_path, map_location=lambda storage, loc: storage)
    print('Loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    state_dict_ = checkpoint['state_dict']
    state_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('network') and not k.startswith('module_list'):
            state_dict[k[8:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print('Skip loading parameter {}, required shape{}, '
                      'loaded shape{}.'.format(
                          k, model_state_dict[k].shape, state_dict[k].shape))
                state_dict[k] = model_state_dict[k]
        else:
            print('Drop parameter {}.'.format(k))
    for k in model_state_dict:
        if not (k in state_dict):
            print('No param {}.'.format(k))
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    # resume optimizer parameters
    if optimizer is not None and resume:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_lr = lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = start_lr
            print('Resumed optimizer with starting learning rate', start_lr)
        else:
            print('No optimizer parameters in checkpoint.')
    if optimizer is not None:
        return model, optimizer, start_epoch
    else:
        return model


def run_model(image_dir, net_save_dir, model_name, model_path):
    """
    This generates activations and saves in net_save_dir
    Input:
    image_dir: Image directory containing .jpg files
    net_save_dir: directory for saving activations
    Activations are saved in format XY.mat where XY is the image file
    XY.mat contains activations for specific layers in with corresponding layer's name
    """
    if model_name == "all":
        for model in models.keys():
            print("==============Start of Model : ", model, "================")
            model_save_dir = os.path.join(net_save_dir, model)
            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)
            execute_model(image_dir, model_save_dir, models[model])
            print("================End of Model : ", model, "================")
    else:
        model = models[model_name]
        model = load_model(model, model_path)
        execute_model(image_dir, net_save_dir, model)
