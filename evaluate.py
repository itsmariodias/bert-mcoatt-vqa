from utils.vqaTools.vqa import VQA
from utils.vqaEvaluation.vqaEval import VQAEval
from utils.data_gen import VQADataGenerator
from models.build import build_model

import joblib
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import json
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt


def generate_result(data_gen, question_ids, model, label2ans, output_file, verbose=True):
    """
    Generate list of predicted answers for each question and store as json file
    """
    def generator_predict(model, data_gen):
        for data, label in data_gen:
            yield model.predict_on_batch(data)

    print("\nGenerating results...")
    json_list = []
    decoder = np.vectorize(lambda x: label2ans[int(x)])
    progbar = tf.keras.utils.Progbar(len(question_ids), verbose=1)
    predict_gen = generator_predict(model, data_gen)

    for i in range(len(data_gen)):
        y_predict = next(predict_gen)
        y_predict = np.argmax(y_predict, axis=-1)
        y_predict_text = decoder(y_predict)

        for j in range(len(y_predict_text)):
            prediction = y_predict_text[j]
            question_id = question_ids[i * data_gen.batch_size + j]
            json_list.append({'answer': prediction, 'question_id': int(question_id)})
            progbar.add(1)

    json.dump(json_list, open(output_file, 'w'))
    if verbose: print(f"Results generated and saved as {output_file}.")


def vqaEval(C):
    """
    Function to calculate VQA score and generate accuracy reports. Adapted from VQA v2.0 evaluation code.
    """
    annFile = os.path.join(C.VAL_DIR, 'v2_mscoco_val2014_annotations.json')
    quesFile = os.path.join(C.VAL_DIR, 'v2_OpenEnded_mscoco_val2014_questions.json')
    resFile = C.JSON_RESULT

    fileTypes = ['accuracy', 'evalQA', 'evalQuesType', 'evalAnsType']
    [accuracyFile, evalQAFile, evalQuesTypeFile, evalAnsTypeFile] = [
        f'{C.OUTPUT_DIR}/{C.MODEL_NAME}_{fileType}.json' for fileType in fileTypes]

    # create vqa object and vqaRes object
    vqa = VQA(annFile, quesFile)
    vqaRes = vqa.loadRes(resFile, quesFile)

    # create vqaEval object by taking vqa and vqaRes
    vqaEval = VQAEval(vqa, vqaRes, n=2)  # n is precision of accuracy (number of places after decimal), default is 2

    # evaluate results
    """
    If you have a list of question ids on which you would like to evaluate your results, pass it as a list to below 
    function
    By default it uses all the question ids in annotation file
    """
    vqaEval.evaluate()

    if C.VERBOSE:
        # print accuracies
        print("\n")
        print("Overall Accuracy is: %.02f\n" % (vqaEval.accuracy['overall']))
        print("Per Question Type Accuracy is the following:")
        for quesType in vqaEval.accuracy['perQuestionType']:
            print("%s : %.02f" % (quesType, vqaEval.accuracy['perQuestionType'][quesType]))
        print("\n")
        print("Per Answer Type Accuracy is the following:")
        for ansType in vqaEval.accuracy['perAnswerType']:
            print("%s : %.02f" % (ansType, vqaEval.accuracy['perAnswerType'][ansType]))
        print("\n")

    # plot accuracy for various question types
    plt.figure()
    plt.bar(range(len(vqaEval.accuracy['perQuestionType'])), vqaEval.accuracy['perQuestionType'].values(),
            align='center')
    plt.xticks(range(len(vqaEval.accuracy['perQuestionType'])), vqaEval.accuracy['perQuestionType'].keys(),
               rotation='0', fontsize=10)
    plt.title('Per Question Type Accuracy', fontsize=10)
    plt.xlabel('Question Types', fontsize=10)
    plt.ylabel('Accuracy', fontsize=10)
    plt.savefig(os.path.join(C.OUTPUT_DIR, f"{C.MODEL_NAME}_{C.RUN_MODE}_qtype_acc.png"))

    if C.VERBOSE: print(f"Per question type accuracy graph saved at {C.OUTPUT_DIR}.")

    # save evaluation results
    json.dump(vqaEval.accuracy, open(accuracyFile, 'w'))
    json.dump(vqaEval.evalQA, open(evalQAFile, 'w'))
    json.dump(vqaEval.evalQuesType, open(evalQuesTypeFile, 'w'))
    json.dump(vqaEval.evalAnsType, open(evalAnsTypeFile, 'w'))

    if C.VERBOSE: print(f"\nEvaluation results saved at {C.OUTPUT_DIR}.")


def evaluate(C, model=None):
    # evaluate the model at end of training

    print("\nStarting evaluation process...")
    eval_gen = VQADataGenerator(C, mode=C.RUN_MODE)

    # If only evaluating, then load model
    if not model:
        model = build_model(C)

    # this is list obtaining corresponding index to each answer
    if C.ANSWERS_TYPE == "softscore":
        label2ans = joblib.load(C.LABEL2ANS)
    else:
        label2ans = joblib.load(C.LABEL_ENCODER_PATH).classes_

    if C.RUN_MODE == 'eval':
        question_ids = pd.DataFrame(json.load(open(C.QUESTION_PATH["val"], 'r'))['questions'])['question_id']
    elif C.RUN_MODE == 'test':
        question_ids = pd.DataFrame(json.load(open(C.QUESTION_PATH["test"], 'r'))['questions'])['question_id']

    generate_result(eval_gen, question_ids, model, label2ans, C.JSON_RESULT, C.VERBOSE)

    if C.RUN_MODE == 'eval':
        if C.VERBOSE: print("\nCalculating score for VQA v2 validation dataset...")
        vqaEval(C)
    else:
        if C.VERBOSE: print("\nTo calculate score for the test-set, check out https://visualqa.org/challenge.html")

    print("Evaluation done.")
