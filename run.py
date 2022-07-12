from configs.base_config import Config, parse_to_dict
from evaluate import evaluate
from train import train
from utils.ans_encode import ans_encode
from utils.coco_extract import coco_extract
from utils.misc import set_seed

import argparse
import os
import yaml


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Training Arguments')

    parser.add_argument('--RUN', dest='RUN_MODE',
                        choices=['train', 'eval', 'test'],
                        help='{train, eval, test}',
                        type=str, required=True)

    parser.add_argument('--CONFIG', dest='CONFIG_NAME',
                        help='configuration for model building located at configs/, e.g. bert_hiecoatt, '
                             'bert_hiecoatt_shared, bert_mcoatt',
                        type=str, required=True)

    parser.add_argument('--SPLIT', dest='TRAIN_SPLIT',
                        choices=['train', 'train+val'],
                        help="set training split, eg.'train', 'train+val'",
                        type=str)

    parser.add_argument('--VAL_MODE', dest='VAL_MODE',
                        choices=['val-half', 'val'],
                        help="set val to full or half, eg. 'val-half', 'val'",
                        type=str)

    parser.add_argument('--EVAL', dest='EVAL',
                        help="evaluate model after training",
                        type=bool)

    parser.add_argument('--PRELOAD', dest='PRELOAD',
                        help='pre-load the image features into memory to increase the I/O speed',
                        type=bool)

    parser.add_argument('--SAVE_PERIOD', dest='SAVE_PERIOD',
                        help='after how many epochs to save checkpoint',
                        type=int)

    parser.add_argument('--TRAIN_STEP', dest='TRAIN_STEP_SIZE',
                        help='how many train steps per epoch',
                        type=int)

    parser.add_argument('--VAL_STEP', dest='VAL_STEP_SIZE',
                        help='how many val steps per epoch',
                        type=int)

    parser.add_argument('--SEED', dest='SEED',
                        help='fix random seed',
                        type=int)

    parser.add_argument('--VERSION', dest='VERSION',
                        help='version control',
                        type=str)

    parser.add_argument('--DATA_DIR', dest='DATA_DIR',
                        help='data root path',
                        type=str)

    parser.add_argument('--OUTPUT_DIR', dest='OUTPUT_DIR',
                        help='output/results root path',
                        type=str)

    parser.add_argument('--CHECKPOINT_DIR', dest='CHECKPOINT_DIR',
                        help='checkpoint root path',
                        type=str)

    parser.add_argument('--FEATURES_DIR', dest='FEATURES_DIR',
                        help='image features root path',
                        type=str)

    parser.add_argument('--START_EPOCH', dest='START_EPOCH',
                        help='initial epoch of model before training/ epoch of model to evaluate on',
                        type=int)

    parser.add_argument('--NUM_EPOCHS', dest='NUM_EPOCHS',
                        help='number of epochs to train model for/',
                        type=int)

    parser.add_argument('--VERBOSE', dest='VERBOSE',
                        help='verbose print',
                        type=bool)

    args = parser.parse_args()
    return args


def run(C):
    set_seed(C.SEED)

    coco_extract(C)

    ans_encode(C)

    if C.RUN_MODE == "train":
        model = train(C)
        if C.EVAL:
            if C.SPLIT == 'train':
                C.RUN_MODE = "eval"
            else:
                C.RUN_MODE = "test"
            evaluate(C, model)
    else:
        evaluate(C)


if __name__ == "__main__":
    print("Loading configuration...")

    C = Config()

    args = parse_args()
    args_dict = parse_to_dict(args)

    config_path = os.path.join("configs", f"{args_dict['CONFIG_NAME']}.yml")
    if os.path.exists(config_path):
        cfg_file = config_path
        with open(cfg_file, 'r') as f:
            yaml_dict = yaml.safe_load(f)
    else:
        print(f"ERROR: Config file does not exist. Given {args_dict['CONFIG_NAME']}.")
        exit(-1)

    args_dict = {**yaml_dict, **args_dict}
    C.add_args(args_dict)
    C.set_paths()
    if 'FEATURES_DIR' in args_dict.keys():
        C.FEATURES_DIR = args_dict['FEATURES_DIR']
    print("Configuration loaded.")

    print("\nRunning program...")
    run(C)

    print("Program done.")
