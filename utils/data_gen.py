from utils.data_utils import tokenize, image_feat_preload, ans_proc, select_top_n_answers, \
    image_feat_path_load, get_answer

import tensorflow as tf
import json
import numpy as np
import pandas as pd
import joblib
import math
from sklearn.model_selection import train_test_split


# data generator to obtain the image and question features along with their masks
# supports shuffling and custom step size during training
class VQADataGenerator(tf.keras.utils.Sequence):

    def __init__(self, C, mode="train"):
        if C.VERBOSE: print(f"\nLoading {mode} data...")

        data, answers = [], []

        self.step_size, self.shuffle, self.labels = None, False, True

        if mode in ['train']:
            for split in C.TRAIN_SPLIT.split('+'):
                # load data from each json file
                data += json.load(open(C.QUESTION_PATH[split], 'r'))['questions']
                # contains list of answers given they appear in the ans vocab for each question
                answers += json.load(open(C.ANSWER_PATH[split], 'r'))['annotations']

            self.step_size, self.shuffle = C.TRAIN_STEP_SIZE, True

        elif mode in ['val']:
            # load data from each json file
            data += json.load(open(C.QUESTION_PATH['val'], 'r'))['questions']
            # contains list of answers given they appear in the ans vocab for each question
            answers += json.load(open(C.ANSWER_PATH['val'], 'r'))['annotations']

            self.step_size = C.VAL_STEP_SIZE

        elif mode in ['eval']:
            # load data from each json file
            data += json.load(open(C.QUESTION_PATH['val'], 'r'))['questions']

            self.labels = False

        elif mode in ['test']:
            # load data from each json file
            data += json.load(open(C.QUESTION_PATH['test'], 'r'))['questions']

            self.labels = False

        else:
            print(f'ERROR: Invalid data mode. Given {mode}.')
            exit(-1)

        data = pd.DataFrame(data)

        # take portion of val split for performing validation during training to speed up training process.
        if mode in ['val'] and C.VAL_MODE == "val-half":
            _, data, _, answers = train_test_split(
                data, answers, test_size=0.30, shuffle=True, random_state=42,
            )

        # extract the data column wise
        questions = data['question'].tolist()
        images = data['image_id'].tolist()

        # load label encoder generated during answer encoding
        label_encoder = joblib.load(C.LABEL_ENCODER_PATH)

        # select the data that corresponds to the answers that appear in the label encoder
        if mode in ['train', 'val', 'val-half']:
            answers = get_answer(answers, C.ANSWERS_TYPE)
            questions, answers, images = select_top_n_answers(questions, answers, images, label_encoder.classes_)

        if C.VERBOSE: print("Dataset size: ", len(questions))

        # tokenize questions using BERT tokenizer
        if C.VERBOSE: print("Tokenizing all questions...")
        self.questions = tokenize(questions, C.QUES_SEQ_LEN, C.BERT_MODEL_PATH['base'])
        if C.VERBOSE: print("Questions loaded.")

        # load images, either their paths or all image features preloaded
        self.image_dict = {}
        if C.PRELOAD:
            if C.VERBOSE: print("Preloading all image features...")
            self.images, self.image_dict = image_feat_preload(images, C.FEATURES_DIR)
        else:
            # obtain the paths to pretrained image features
            self.images = image_feat_path_load(images, C.FEATURES_DIR)

        self.img_seq_len = C.IMG_SEQ_LEN
        if C.VERBOSE: print("Images loaded.")

        # Encode answers if working on train or val splits
        if mode in ['train', 'val', 'val-half']:
            if C.VERBOSE: print("Encoding all answers...")
            self.answers = ans_proc(answers, label_encoder, C.NUM_CLASSES)
            if C.VERBOSE: print("Answers loaded.")

        self.batch_size = C.BATCH_SIZE
        self.indices = np.arange(len(self.questions['input_ids']))
        if C.VERBOSE: print("Dataset ready.")

    def __len__(self):
        if self.step_size:
            return self.step_size
        return math.ceil(len(self.questions['input_ids']) / self.batch_size)

    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        question_ids = self.questions['input_ids'][inds]
        question_masks = self.questions['attention_mask'][inds]

        img_feats = self.get_images(self.images[inds])

        ans_batch = None
        if self.labels:
            ans_batch = self.answers[inds]

        return [img_feats, question_ids, question_masks], ans_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    # load the images
    def load_feat(self, x):
        if self.image_dict:
            arr = self.image_dict[x]
        else:
            arr = np.load(x)
        return arr

    def get_images(self, images_input):
        arr = np.array(list(map(self.load_feat, images_input)))
        return np.reshape(arr, (-1, arr.shape[2], arr.shape[3]))
