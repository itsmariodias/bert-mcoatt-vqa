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
        self.TRAIN_SPLIT = "train"  # ["train", "train+val"]
        self.VAL_MODE = "val-half"  # ["val-half", "val"]
        self.PRELOAD = False
        self.BATCH_SIZE = 64
        self.SAVE_PERIOD = 1
        self.TRAIN_STEP_SIZE = None
        self.VAL_STEP_SIZE = None
        self.VERSION = self.SEED
        self.EVAL = True

        # -- INPUT PARAMETERS --
        self.IMG_SEQ_LEN = 196
        self.IMG_EMBED_SIZE = 512
        self.QUES_SEQ_LEN = 14
        self.FEATURES_TYPE = "resnet152"  # ["vgg19", "resnet152"]

        # -- OUTPUT PARAMETERS --
        self.NUM_CLASSES = 1000
        self.ANSWERS_TYPE = "mcq"  # ["mcq", "modal"]

        # -- MODEL HYPERPARAMETERS --
        self.DROPOUT_RATE = 0.1
        self.HIDDEN_SIZE = 512
        self.NUM_LAYERS = 12
        self.NUM_HEADS = 12

        # -- OPTIMIZER VARIABLES -- 
        self.BASE_LEARNING_RATE = 1e-4
        self.ADAM_BETA_1 = 0.9
        self.ADAM_BETA_2 = 0.999
        self.ADAM_EPSILON = 1e-7

        # -- GLOBAL PATHS ---
        self.DATA_DIR = "data"
        self.OUTPUT_DIR = "results"
        self.CHECKPOINT_DIR = "checkpoints"
        self.MODEL_NAME = None
        self.TRAIN_DIR = None
        self.VAL_DIR = None
        self.TEST_DIR = None
        self.FEATURES_DIR = None
        self.LOG_PATH = None
        self.CHECKPOINT_PATH = None
        self.JSON_RESULT = None
        self.LABEL_ENCODER_PATH = None
        self.QUESTION_PATH = None
        self.ANSWER_PATH = None
        self.BERT_MODEL_PATH = None

        self.set_paths()

        # -- CHECKPOINT TRAINING --
        self.START_EPOCH = 0
        self.NUM_EPOCHS = 10

    def set_paths(self):
        self.MODEL_NAME = f'{self.CONFIG_NAME}_{self.VERSION}'

        self.TRAIN_DIR = os.path.join(self.DATA_DIR, "train")
        self.VAL_DIR = os.path.join(self.DATA_DIR, "val")
        self.TEST_DIR = os.path.join(self.DATA_DIR, "test")

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

        self.QUESTION_PATH = {
            'train': os.path.join(self.TRAIN_DIR, "v2_OpenEnded_mscoco_train2014_questions.json"),
            'val': os.path.join(self.VAL_DIR, "v2_OpenEnded_mscoco_val2014_questions.json"),
            'test': os.path.join(self.TEST_DIR, "v2_OpenEnded_mscoco_test2015_questions.json"),
        }
        self.ANSWER_PATH = {
            'train': os.path.join(self.TRAIN_DIR, "v2_mscoco_train2014_annotations.json"),
            'val': os.path.join(self.VAL_DIR, "v2_mscoco_val2014_annotations.json"),
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
