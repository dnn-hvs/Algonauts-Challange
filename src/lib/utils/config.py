import argparse
from datetime import datetime
import os
import lib.utils.networks_factory as networks_factory
import lib.utils.constants as constants
import torch


class Config(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description='generate DNN activations from a stimuli dir')

        self.parser.add_argument('--fullblown', action="store_true",
                                 help='This will run all combinations of tasks and image sets on the \
                                    models given in --models_dir. This will genrate features, \
                                    Create RDMs and Evaluate the results.')

        self.parser.add_argument('--generate_features', action="store_true",
                                 help='This will run all combinations of tasks and image sets on the \
                                    models given in --models_dir. This will only genrate features.')

        self.parser.add_argument('--create_rdms', action="store_true",
                                 help='This will run all combinations of tasks and image sets on the \
                                    models given in --models_dir. This will only create RDMs.')

        self.parser.add_argument('--evaluate_results', action="store_true",
                                 help='This will run all combinations of tasks and image sets on the \
                                    models given in --models_dir. This will only evaluate results.')

        self.parser.add_argument("--models_dir",
                                 help='Path to the directory that contains all the best models of all architectures',
                                 default=None)

        self.parser.add_argument('--feat_dir',
                                 help='Features directory path', default="./feats", type=str)

        self.parser.add_argument(
            '--rdms_dir', help='RDM directory path', default="./rdms", type=str)

        RDM_distance_choice = ['pearson', 'kernel']

        self.parser.add_argument('-d', '--distance', help='distance for RDMs',
                                 default="pearson", choices=RDM_distance_choice)

        self.parser.add_argument('--gpus', default='0',
                                 help='-1 for CPU, use comma for multiple gpus')

        self.parser.add_argument('--task', default='all',
                                 help='fmri | meg | all')

        self.parser.add_argument('--image_set', default='all',
                                 help='92 | 118 | 78 | all')

        self.parser.add_argument("--arch", help='DNN choice',
                                 default="all", choices=networks_factory.models.keys())

        self.parser.add_argument("--exp", help='Experiment name')
        self.parser.add_argument("--load_model",
                                 help='Path to the desired model to be tested of the architecture specified',
                                 default=None)

        # Directories
        self.parser.add_argument('-id', '--image_dir', help='stimulus directory path',
                                 default=None, type=str)
        self.parser.add_argument('--exp_id', help='Stores feats, rdms and results in this directory',
                                 default=None, type=str)

        self.parser.add_argument('--test_set', help='perform the operations only on the test data',
                                 action="store_true")

    def parse(self, args=''):
        if args == '':
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)

        opt.gpus_str = opt.gpus
        opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
        opt.gpus = [i for i in range(
            len(opt.gpus))] if opt.gpus[0] >= 0 else [-1]

        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
        opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

        if opt.fullblown or opt.generate_features or opt.create_rdms or opt.evaluate_results:
            if opt.test_set:
                opt.image_sets = ['78']

            else:
                opt.image_sets = ['92', '118']

            if opt.exp_id is not None:
                opt.feat_dir = os.path.join(opt.exp_id, "feats")
                opt.rdms_dir = os.path.join(opt.exp_id, "rdms")
            return opt

        if opt.image_set == "all":
            opt.feat_save_dir = opt.feat_dir

        else:
            opt.feat_save_dir = os.path.join(
                opt.feat_dir, opt.image_set+"_feats")


##########################################################
        if opt.image_dir != None:
            image_dir_name = opt.image_dir.split("/")[-1]
            feat_save_dir = os.path.join(opt.feat_dir, image_dir_name+"_feats")

        if opt.arch != 'all':
            net_save_dir = os.path.join(feat_save_dir, opt.exp_name)
        else:
            net_save_dir = feat_save_dir

        if not os.path.exists(net_save_dir):
            os.makedirs(net_save_dir)

        opt.net_save_dir = net_save_dir

        if opt.image_dir is None:

            opt.image_dir = "../data/Training_Data/" + \
                opt.image_set+"_Image_Set/"+opt.image_set+"images"

        # opt.save_dir = os.path.join('../models', opt.arch + "_" +
        #                             str(datetime.now().strftime(
        #                                 "%d-%b-%y--%X")))
        return opt

    def init(self, args=''):
        opt = self.parse(args)
        return opt
