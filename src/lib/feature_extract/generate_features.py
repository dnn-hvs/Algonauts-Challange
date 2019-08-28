import argparse
import os
# functions to generate features in utils.py
from lib.utils import utils, config, networks_factory, constants
import torch
import glob
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from collections import OrderedDict
from torch.autograd import Variable
import scipy.io as sio


class GenerateFeatures():
    def __init__(self, config):
        self.config = config

    def execute_model(self, model, feats_save_dir):
        if torch.cuda.is_available():
            model.to(self.config.device)
        model.eval()
        centre_crop = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image_list = glob.glob(self.config.image_dir + "/*.jpg")
        print(self.config.image_dir)
        image_list.sort()
        image_list = image_list
        print(len(image_list))
        for image in tqdm(image_list):
            img = Image.open(image)
            filename = image.split("/")[-1].split(".")[0]
            input_img = Variable(centre_crop(img).unsqueeze(0))
            if torch.cuda.is_available():
                input_img = input_img.to(self.config.device)

            x = model.forward(input_img)
            save_path = os.path.join(feats_save_dir, filename+".mat")
            feats = OrderedDict()
            for key, value in x.items():
                feats[key] = value.data.cpu().numpy()
            sio.savemat(save_path, feats)
        print(str(feats.keys()))

    def get_model(self, model, load_model):
        if self.config.load_model is not None or self.config.fullblown:
            model = utils.get_model(model)
            return utils.load_model(model, load_model)
        else:
            return utils.get_model(self.config.arch)

    def run(self):
        if self.config.fullblown:
            for image_set in self.config.image_sets:
                print("Image Set: ", image_set)
                self.config.image_dir = os.path.join("../data/Training_Data/" +
                                                     image_set+"_Image_Set", image_set+"images")
                models_list = glob.glob(self.config.models_dir + "/*.pth")
                for model_pth in models_list:
                    pth_name = model_pth.split(constants.FORWARD_SLASH)[-1]
                    model_name = pth_name.split("_")[0]
                    subdir_name = pth_name.split(".")[0]
                    print("model_name: ", model_name,
                          " subdir_name: ", subdir_name)
                    path = os.path.join(
                        self.config.feat_dir, image_set+"images_feats", subdir_name)
                    if not os.path.exists(path):
                        os.makedirs(path)
                    model = self.get_model(model_name, model_pth)
                    self.execute_model(model, path)
            return

        # if self.config.image_set == "all":
        #     for image_set in ['92', '118']:
        #         # Images Direcoty
        #         self.config.image_dir = "../data/Training_Data/" + \
        #             image_set+"_Image_Set"+image_set+"images"

        #         if self.config.models_dir is not None:
        #             models_list = glob.glob(self.config.models_dir + "/*.pth")
        #             for model_pth in models_list:
        #                 model_name = model_pth.split[constants.FORWARD_SLASH][-1].split("_")[
        #                     0]
        #                 model_save_dir = os.path.join(
        #                     self.config.net_save_dir, model_name)
        #                 if not os.path.exists(model_save_dir):
        #                     os.makedirs(model_save_dir)

        #                 model = self.get_model(model, path)
        #                 self.execute_model(model, model_save_dir)

        # if self.config.arch == "all":
        #     for model in models.keys():
        #         print("==============Start of Model : ",
        #               model, "================")

        #         if not os.path.exists(model_save_dir):
        #             os.makedirs(model_save_dir)

        #         model = self.get_model(model, path)
        #         self.execute_model(model, model_save_dir)
        #         print("================End of Model : ",
        #               model, "================")
        # else:
        #     model = models[self.config.arch]
        #     model_save_dir = os.path.join(
        #         self.config.net_save_dir, self.config.arch)

        #     model = self.get_model(model, path)
        #     self.execute_model(model, model_save_dir)
