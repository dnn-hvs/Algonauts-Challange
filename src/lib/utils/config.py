import argparse
from datetime import datetime
import os
import networks_factory


class Config(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description='generate DNN activations from a stimuli dir')
        self.parser.add_argument('-id', '--image_dir', help='stimulus directory path',
                                 default="../data/Training_Data/118_Image_Set/118images", type=str)
        self.parser.add_argument('-sd', '--feat_dir',
                                 help='Features directory path', default="./feats", type=str)
        self.parser.add_argument(
            '-sd', '--rdms_dir', help='RDM directory path', default="./rdms", type=str)

        self.parser.add_argument("--arch", help='DNN choice',
                                 default="all", choices=networks_factory.models.keys())
        self.parser.add_argument("--exp", help='Experiment name')
        self.parser.add_argument("--load_model",
                                 help='Path to the desired model to be tested of the architecture specified',
                                 default=None)

        RDM_distance_choice = ['pearson']

        self.parser.add_argument('-d', '--distance', help='distance for RDMs',
                                 default="pearson", choices=RDM_distance_choice)

    def parse(self, args=''):
        if args == '':
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)

        image_dir_name = opt.image_dir.split("/")[-1]
        feat_save_dir = os.path.join(opt.feat_dir, image_dir_name+"_feats")

        if opt.arch != 'all':
            net_save_dir = os.path.join(feat_save_dir, opt.exp_name)
        else:
            net_save_dir = feat_save_dir

        if not os.path.exists(net_save_dir):
            os.makedirs(net_save_dir)

        opt.net_save_dir = net_save_dir

        # opt.save_dir = os.path.join('../models', opt.arch + "_" +
        #                             str(datetime.now().strftime(
        #                                 "%d-%b-%y--%X")))
        return opt

    def init(self, args=''):
        opt = self.parse(args)
        return opt
