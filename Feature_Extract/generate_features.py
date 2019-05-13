import argparse
import os
# functions to generate features in utils.py
from utils import *


def generate_features(image_dir, save_dir, net):

    image_dir_name = image_dir.split("/")[-1]
    feat_save_dir = os.path.join(save_dir, image_dir_name+"_feats")
    net_save_dir = os.path.join(feat_save_dir, net)
    if not os.path.exists(net_save_dir):
        os.makedirs(net_save_dir)
    run_model(image_dir, net_save_dir, net)


def main():
    # dnns list
    dnns_list = ['alexnet', 'vgg', 'resnet', 'sqnet1_0', 'sqnet1_1', ]

    # ArgumentParser
    parser = argparse.ArgumentParser(
        description='generate DNN activations from a stimuli dir')
    parser.add_argument('-id', '--image_dir', help='stimulus directory path',
                        default="/media/kshitid20/My Passport1/Algonauts/algonautsChallenge2019/118_Image_Set/118images", type=str)
    parser.add_argument('-sd', '--save_dir',
                        help='save directory path', default="./feats", type=str)
    parser.add_argument("--net", help='DNN choice',
                        default="vgg", choices=dnns_list)
    args = vars(parser.parse_args())

    image_dir = args['image_dir']
    save_dir = args['save_dir']
    net = args['net']

    # generate features/activations
    generate_features(image_dir, save_dir, net)


if __name__ == "__main__":
    main()
