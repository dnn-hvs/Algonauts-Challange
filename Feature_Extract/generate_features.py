import argparse
import os
# functions to generate features in utils.py
from utils import *


def generate_features(image_dir, save_dir, net, path):

    image_dir_name = image_dir.split("/")[-1]
    feat_save_dir = os.path.join(save_dir, image_dir_name+"_feats")
    if net != 'all':
        net_save_dir = os.path.join(feat_save_dir, net)
    else:
        net_save_dir = feat_save_dir

    if not os.path.exists(net_save_dir):
        os.makedirs(net_save_dir)
    run_model(image_dir, net_save_dir, net, path)


def main():
    # dnns list
    dnns_list = ['alexnet', 'vgg', 'resnet50',
                 'sqnet1_0', 'sqnet1_1', 'densenet201', 'googlenet', 'inception', 'all']

    # ArgumentParser
    parser = argparse.ArgumentParser(
        description='generate DNN activations from a stimuli dir')
    parser.add_argument('-id', '--image_dir', help='stimulus directory path',
                        default="/media/kshitid20/My Passport1/Algonauts/algonautsChallenge2019/118_Image_Set/118images", type=str)
    parser.add_argument('-sd', '--save_dir',
                        help='save directory path', default="./feats", type=str)
    parser.add_argument("--net", help='DNN choice',
                        default="all", choices=dnns_list)
    parser.add_argument("--load_model",
                        help='Path to the desired model to be tested of the architecture specified',
                        default=".")

    args = vars(parser.parse_args())

    image_dir = args['image_dir']
    save_dir = args['save_dir']
    net = args['net']
    path = args['load_model']

    # generate features/activations
    generate_features(image_dir, save_dir, net, path)


if __name__ == "__main__":
    main()
