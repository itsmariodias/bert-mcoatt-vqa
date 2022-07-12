"""
Utility function to extract and encode the answers from the train split.
Each question in the VQA dataset contains 10 human annotated answers.
We extract a single answer from each question and then select the top N most occurring answers.
Then we encode each answer, so we can one-hot encode them during training.

The default answer types we use are 'mcq' and 'modal'.
'mcq' extracts the machine generated answer.
'modal' extracts the most common answer from the 10 human annotated answers.
"""

from configs.base_config import Config
from utils.data_utils import get_answer

from sklearn.preprocessing import LabelEncoder
import json
from collections import Counter
import joblib
import os


def ans_encode(C):
    if os.path.exists(C.LABEL_ENCODER_PATH):
        if C.VERBOSE: print(f"\nAll {C.ANSWERS_TYPE} answers have already been encoded. Skipping...")
        return

    if C.VERBOSE: print(f"\nAll {C.ANSWERS_TYPE} answers have not been encoded.")
    print("Beginning encoding process...")

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

    print(f"All {C.ANSWERS_TYPE} answers have been successfully encoded.")


if __name__ == "__main__":
    C = Config()
    ans_encode(C)
