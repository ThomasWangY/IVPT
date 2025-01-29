"""
From: https://github.com/zxhuang1698/interpretability-by-parts/blob/master/src/cub200/eval_interp.py
"""
# pytorch & misc
import torch
import torchvision.transforms as transforms
from data_sets import FineGrainedBirdClassificationParts
from load_model import load_model_ivpt
import argparse
import copy
from engine.eval_interpretability_nmi_ari_keypoint import eval_nmi_ari, eval_kpr
from engine.eval_fg_bg import FgBgIoU
from utils.training_utils.engine_utils import load_state_dict_ivpt

import os
import torch
import argparse
from eval.eval_interpretability import evaluate_consistency, evaluate_stability

torch.multiprocessing.set_sharing_strategy('file_system')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate model interpretability via part parsing quality'
    )
    parser.add_argument('--model_arch', default='resnet50', type=str,
                        help='pick model architecture')
    parser.add_argument('--use_torchvision_resnet_model', default=False, action='store_true')

    # Data
    parser.add_argument('--data_path',
                        help='directory that contains cub files, must'
                             'contain folder "./images"', required=True)
    parser.add_argument('--image_sub_path', default='images', type=str, required=False)
    parser.add_argument('--dataset', default='cub', type=str)
    parser.add_argument('--anno_path_test', default='', type=str, required=False)
    parser.add_argument('--center_crop', default=False, action='store_true')

    # Eval mode
    parser.add_argument('--eval_mode', default='keypoint', choices=['keypoint', 'nmi_ari', 'fg_bg'], type=str)

    # Model params
    parser.add_argument('--num_parts', help='number of parts to predict',
                        default=8, type=int)
    parser.add_argument('--n_pro', default="17,14,11,8,5", type=str)
    parser.add_argument('--image_size', default=448, type=int)
    parser.add_argument('--output_stride', default=32, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=1, type=int)
    # Modulation
    parser.add_argument('--modulation_type', default="original",
                        choices=["original", "layer_norm", "parallel_mlp", "parallel_mlp_no_bias",
                                 "parallel_mlp_no_act", "parallel_mlp_no_act_no_bias", "none"],
                        type=str)
    parser.add_argument('--modulation_orth', default=False, action='store_true',
                        help='use orthogonality loss on modulated features')
    # Part Dropout
    parser.add_argument('--part_dropout', default=0.0, type=float)

    # Add noise to vit output features
    parser.add_argument('--noise_variance', default=0.0, type=float)

    # Gumbel Softmax
    parser.add_argument('--gumbel_softmax', default=False, action='store_true')
    parser.add_argument('--gumbel_softmax_temperature', default=1.0, type=float)
    parser.add_argument('--gumbel_softmax_hard', default=False, action='store_true')

    # Model path
    parser.add_argument('--model_path', default=None, type=str)

    # Classifier type
    parser.add_argument('--classifier_type', default="linear",
                        choices=["linear", "independent_mlp"], type=str)

    # New Proto
    parser.add_argument('--gpuid', type=str, default='0')
    parser.add_argument('--data_set', default='CUB2011', type=str)
    parser.add_argument('--nb_classes', type=int, default=200)
    parser.add_argument('--test_batch_size', type=int, default=30)
    parser.add_argument('--half_size', type=int, default=84)

    # Model
    parser.add_argument('--base_architecture', type=str, default='vgg16')
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--prototype_shape', nargs='+', type=int, default=[2000, 64, 1, 1])
    parser.add_argument('--prototype_activation_function', type=str, default='log')
    parser.add_argument('--add_on_layers_type', type=str, default='regular')
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--resume', type=str)

    args = parser.parse_args()
    return args


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_cls = 200

    # Add arguments to args
    args.eval_only = True
    args.pretrained_start_weights = True

    # Load the model
    net = load_model_ivpt(args, num_cls)

    net = net.cuda()

    snapshot_data = torch.load(args.model_path, map_location=torch.device('cpu'))
    if 'model_state' in snapshot_data:
        _, state_dict = load_state_dict_ivpt(snapshot_data)
    else:
        state_dict = copy.deepcopy(snapshot_data)
    net.load_state_dict(state_dict, strict=True)
    net.eval()
    net.to(device)

    consistency_score = evaluate_consistency(net, args, half_size=args.half_size)
    print('Consistency Score : {:.2f}%'.format(consistency_score))

    stability_score = evaluate_stability(net, args, half_size=args.half_size)
    print('Stability Score : {:.2f}%'.format(stability_score))


if __name__ == '__main__':
    arguments = parse_args()
    main(arguments)
