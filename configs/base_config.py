import os
import random
from types import MethodType


def parse_to_dict(args):
    args_dict = {}
    for arg in dir(args):
        if not arg.startswith('_') and not isinstance(getattr(args, arg), MethodType):
            if getattr(args, arg) is not None:
                args_dict[arg] = getattr(args, arg)

    return args_dict


class Config():
    def __init__(self):
        # CONFIGURATION
        self.CONFIG_NAME = None

        # Seed for consistency
        self.SEED = random.randint(0, 99999999)

        # Verbosity
        self.VERBOSE = False

        # -- TRAINING PARAMETERS --
        self.RUN_MODE = "train"  # ["train", "eval", "test"]
        self.MODEL_TYPE = "bert_hiecoatt"  # ["bert_hiecoatt", "bert_hiecoatt_shared", "bert_mcoatt"]
        self.TRAIN_SPLIT = "train"  # ["train", "train+val", "train+val+vg"]
        self.VAL_MODE = "val-half"  # ["val-half", "val"]
        self.PRELOAD = False
        self.BATCH_SIZE = 64
        self.SAVE_PERIOD = 1
        self.TRAIN_STEP_SIZE = None
        self.VAL_STEP_SIZE = None
        self.VERSION = self.SEED
        self.EVAL = True

        # -- INPUT PARAMETERS --
        self.IMG_SEQ_LEN = 196  # [196, 100, 36]
        self.IMG_EMBED_SIZE = 512
        self.QUES_SEQ_LEN = 14
        self.FEATURES_TYPE = "resnet152"  # ["vgg19", "resnet152", "bottom_up_100", "bottom_up_36"]
        self.PRETRAIN = True

        # -- OUTPUT PARAMETERS --
        self.NUM_CLASSES = 1000  # [1000, 3000, 3129]
        self.ANSWERS_TYPE = "mcq"  # ["mcq", "modal", "softscore"]

        # -- MODEL HYPER-PARAMETERS --
        self.DROPOUT_RATE = 0.1
        self.HIDDEN_SIZE = 768  # [512, 768, 1024]
        self.FLAT_MLP_SIZE = 512
        self.FLAT_OUT_SIZE = 1024  # [1024, 1536, 2048]
        self.ACTIVATION = "gelu"  # ["relu", "gelu", "gelu_new", "silu"]
        self.LAYER_NORM_EPSILON = 1e-12
        self.NUM_LAYERS = 12  # [2, 4, 6, 8, 12, 24]
        self.NUM_HEADS = 12  # [8, 12, 16]
        self.FF_SIZE = self.HIDDEN_SIZE * 4
        self.INIT_RANGE = 0.02

        # -- OPTIMIZER VARIABLES -- 
        self.BASE_LEARNING_RATE = 1e-4  # [2e-5, 3e-5, 5e-5, 1e-4]
        self.ADAM_BETA_1 = 0.9
        self.ADAM_BETA_2 = 0.999  # [0.98, 0.999]
        self.ADAM_EPSILON = 1e-7  # [1e-7, 1e-9]
        self.ADAM_WEIGHT_DECAY = 0  # [0, 0.01]
        self.CURRENT_LEARNING_RATE = self.BASE_LEARNING_RATE  # used if resuming training with weight decay

        # -- GLOBAL PATHS ---
        self.DATA_DIR = "data"
        self.OUTPUT_DIR = "results"
        self.CHECKPOINT_DIR = "checkpoints"
        self.MODEL_NAME = None
        self.TRAIN_DIR = None
        self.VAL_DIR = None
        self.TEST_DIR = None
        self.VG_DIR = None
        self.FEATURES_DIR = None
        self.LOG_PATH = None
        self.CHECKPOINT_PATH = None
        self.JSON_RESULT = None
        self.LABEL_ENCODER_PATH = None
        self.QUESTION_PATH = None
        self.ANSWER_PATH = None
        self.ANSWER_TARGET_PATH = None
        self.BERT_MODEL_PATH = None
        self.ANS2LABEL = None
        self.LABEL2ANS = None

        self.set_paths()

        # -- CHECKPOINT TRAINING --
        self.START_EPOCH = 0
        self.NUM_EPOCHS = 10

    def set_paths(self):
        self.MODEL_NAME = f'{self.CONFIG_NAME}_{self.VERSION}'

        self.TRAIN_DIR = os.path.join(self.DATA_DIR, "train")
        self.VAL_DIR = os.path.join(self.DATA_DIR, "val")
        self.TEST_DIR = os.path.join(self.DATA_DIR, "test")
        self.VG_DIR = os.path.join(self.DATA_DIR, "vg")

        self.FEATURES_DIR = os.path.join(self.DATA_DIR, self.FEATURES_TYPE)

        self.LOG_PATH = os.path.join(
            self.OUTPUT_DIR, self.MODEL_NAME + "_log.csv"
        )
        self.CHECKPOINT_PATH = os.path.join(
            self.CHECKPOINT_DIR, self.MODEL_NAME + "_epoch_{epoch:02d}.h5"
        )
        self.JSON_RESULT = os.path.join(
            self.OUTPUT_DIR, self.MODEL_NAME + "_results.json"
        )
        self.LABEL_ENCODER_PATH = os.path.join(self.DATA_DIR,
                                               f'labelencoder_{self.NUM_CLASSES}_{self.ANSWERS_TYPE}.pkl')

        self.ANS2LABEL = os.path.join(self.DATA_DIR, 'trainval_ans2label.pkl')
        self.LABEL2ANS = os.path.join(self.DATA_DIR, 'trainval_label2ans.pkl')

        self.QUESTION_PATH = {
            'train': os.path.join(self.TRAIN_DIR, "v2_OpenEnded_mscoco_train2014_questions.json"),
            'val': os.path.join(self.VAL_DIR, "v2_OpenEnded_mscoco_val2014_questions.json"),
            'test': os.path.join(self.TEST_DIR, "v2_OpenEnded_mscoco_test2015_questions.json"),
            'vg': os.path.join(self.VG_DIR, "VG_questions.json"),
        }
        self.ANSWER_PATH = {
            'train': os.path.join(self.TRAIN_DIR, "v2_mscoco_train2014_annotations.json"),
            'val': os.path.join(self.VAL_DIR, "v2_mscoco_val2014_annotations.json"),
            'vg': os.path.join(self.VG_DIR, "VG_annotations.json"),
        }
        self.ANSWER_TARGET_PATH = {
            'train': os.path.join(self.TRAIN_DIR, "train_target.pkl"),
            'val': os.path.join(self.VAL_DIR, "val_target.pkl"),
            'vg': os.path.join(self.VG_DIR, "vg_target.pkl"),
        }
        self.BERT_MODEL_PATH = {
            'base': 'bert-base-uncased',
            'target': f'google/bert_uncased_L-{self.NUM_LAYERS}_H-{self.HIDDEN_SIZE}_A-{self.NUM_HEADS}',
        }

    def add_args(self, args_dict):
        for arg in args_dict:
            setattr(self, arg, args_dict[arg])


if __name__ == "__main__":
    C = Config()
