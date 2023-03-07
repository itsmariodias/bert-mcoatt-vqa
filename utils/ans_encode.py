"""
Utility function to extract and encode the answers from the train split.
Each question in the VQA dataset contains 10 human annotated answers.
We extract a single answer from each question and then select the top N most occurring answers.
Then we encode each answer, so we can one-hot encode them during training.

In case of 'softscore' answer types, we refer to the implementation based on findings in
Tips and Tricks for Visual Question Answering: Learnings from the 2017 Challenge where we assign soft scores to each
annotated answer for each question. The answers we select for encoding are the answers that occur more than 8 times in
the combined train+val VQA v2.0 dataset
Ref: https://arxiv.org/abs/1708.02711

The default answer types we use are 'mcq' and 'modal'.
'mcq' extracts the machine generated answer.
'modal' extracts the most common answer from the 10 human annotated answers.
'softscore' extracts the all answers for each question which occur more than 8 times in the train+val split and assigns
a score between 0 and 1 for each (this score is based on the VQA evaluation metric)
"""

from configs.base_config import Config
from utils.data_utils import get_answer
from utils import compute_softscore as cs

from sklearn.preprocessing import LabelEncoder
import json
from collections import Counter
import joblib
import os


def encode_multiclass(C):
    answers = []

    for split in C.TRAIN_SPLIT.split('+'):
        # contains list of answers given they appear in the ans vocab for each question
        answers += json.load(open(C.ANSWER_PATH[split], 'r'))['annotations']

    answers = get_answer(answers, C.ANSWERS_TYPE)
    if C.VERBOSE: print("Answers loaded.")

    # build a dictionary of answers
    answer_fq = Counter(answers)

    # select top most common answers
    top_answers = [k for k, n in answer_fq.most_common(C.NUM_CLASSES)]

    # encode answers to generate indices for each answer.
    if C.VERBOSE: print(f"Encoding top {C.NUM_CLASSES} answers...")
    label_encoder = LabelEncoder()
    label_encoder.fit(top_answers)

    joblib.dump(label_encoder, C.LABEL_ENCODER_PATH)
    if C.VERBOSE: print(f"Label encoder saved at {C.LABEL_ENCODER_PATH}")


def encode_softscore(C):
    train_answers = json.load(open(C.ANSWER_PATH['train']))['annotations']
    val_answers = json.load(open(C.ANSWER_PATH['val']))['annotations']
    if C.VERBOSE: print("Answers loaded.")

    # Combine the list of train and val answers and filter to select answers which occur more than 8 times (> 9)
    answers = train_answers + val_answers
    occurrence = cs.filter_answers(answers, 9)
    if C.VERBOSE: print('Num of answers that appear >= %d times: %d' % (9, len(occurrence)))

    # encode answers to generate indices for each answer.
    ans2label = cs.create_ans2label(occurrence, 'trainval', cache_root=C.DATA_DIR)
    if C.VERBOSE: print(f"Label encoder and decoder saved at {C.ANS2LABEL} and {C.LABEL2ANS}")

    # apply the above encoder to each data split
    cs.compute_target(train_answers, ans2label, 'train', cache_root=C.TRAIN_DIR)
    cs.compute_target(val_answers, ans2label, 'val', cache_root=C.VAL_DIR)

    if 'vg' in C.TRAIN_SPLIT.split('+'):
        vg_answers = json.load(open(C.ANSWER_PATH['vg']))['annotations']
        cs.compute_target(vg_answers, ans2label, 'vg', cache_root=C.VG_DIR)

    if C.VERBOSE: print(f"Soft scores saved as e.g. {C.DATA_DIR}/dataset/dataset_target.pkl")


def ans_encode(C):
    if (os.path.exists(C.LABEL_ENCODER_PATH) and C.ANSWERS_TYPE != "softscore") or \
            (os.path.exists(C.ANS2LABEL) and os.path.exists(C.LABEL2ANS) and C.ANSWERS_TYPE == "softscore"):
        if C.VERBOSE: print(f"\nAll {C.ANSWERS_TYPE} answers have already been encoded. Skipping...")
        return

    if C.VERBOSE: print(f"\nAll {C.ANSWERS_TYPE} answers have not been encoded.")
    print("Beginning encoding process...")
    if C.ANSWERS_TYPE == "softscore":
        encode_softscore(C)
    elif C.ANSWERS_TYPE in ["mcq", "modal"]:
        encode_multiclass(C)
    else:
        print(f"ERROR: Invalid answer type. Given {C.ANSWERS_TYPE}.")
        exit(-1)

    print(f"All {C.ANSWERS_TYPE} answers have been successfully encoded.")


if __name__ == "__main__":
    C = Config()
    ans_encode(C)
