import argparse
import os
# functions to generate features in utils.py
from utils import utils, config, networks_factory
import torch
import glob
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from collections import OrderedDict
from torch.autograd import Variable
import scipy.io as sio


class GenerateFeature():
    def __init__(self, config):
        self.config = config

    def execute_model(self, model):
        if torch.cuda.is_available():
            model.cuda()
        model.eval()
        centre_crop = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image_list = glob.glob(self.config.image_dir + "/*.jpg")
        image_list.sort()
        image_list = image_list
        print(len(image_list))
        for image in tqdm(image_list):
            img = Image.open(image)
            filename = image.split("/")[-1].split(".")[0]
            input_img = Variable(centre_crop(img).unsqueeze(0))
            if torch.cuda.is_available():
                input_img = input_img.cuda()
            x = model.forward(input_img)
            save_path = os.path.join(self.config.net_save_dir, filename+".mat")
            feats = OrderedDict()
            for key, value in x.items():
                feats[key] = value.data.cpu().numpy()
            sio.savemat(save_path, feats)
        print(str(feats.keys()))

    def run_model(self):
        models = networks_factory.models
        if self.config.arch == "all":
            for model in models.keys():
                print("==============Start of Model : ",
                      model, "================")
                model_save_dir = os.path.join(self.config.net_save_dir, model)
                if not os.path.exists(model_save_dir):
                    os.makedirs(model_save_dir)
                self.execute_model(models[model])
                print("================End of Model : ",
                      model, "================")
        else:
            model = models[self.config.arch]
            if self.config.load_model is not None:
                model = utils.load_model(model, self.config.load_model)
            else:
                model = utils.get_model(self.config.arch)
            self.execute_model(model)
