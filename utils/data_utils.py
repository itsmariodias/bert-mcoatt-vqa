import pandas as pd
import transformers
import tensorflow as tf
import numpy as np
import os
import operator


def select_top_n_answers(questions, answers, images, top_answers):
    """
    Select those questions whose answers appear in the top_answers.
    """
    new_answers = []
    new_questions = []
    new_images = []

    # only those answer which appear in the top_answers are used for training
    for answer, question, image in zip(answers, questions, images):
        if answer in top_answers:
            new_answers.append(answer)
            new_questions.append(question)
            new_images.append(image)

    return new_questions, new_answers, new_images


def tokenize(questions, ques_seq_len, bert_model_path):
    # Tokenize the questions based on bert wordpiece tokenizer
    tokenizer = transformers.BertTokenizer.from_pretrained(bert_model_path)

    # we pad to the max length and truncate questions longer that the max length, adding the [CLS]
    # and [SEP] tokens to the start and end positions and convert tokens to their respective ids
    questions = tokenizer(
        text=questions, 
        padding='max_length', 
        max_length=ques_seq_len, 
        return_tensors='np', 
        truncation=True, 
        return_token_type_ids=False, 
        return_attention_mask=True
    )

    return questions


def image_feat_preload(images, features_dir):
    """
    Preload all image features and store into a dictionary.
    """
    image_dict = {}
    unique_images = set(images)
    progbar = tf.keras.utils.Progbar(len(unique_images), verbose=1)

    for image_id in unique_images:
        image_dict[image_id] = np.load(os.path.join(features_dir, str(image_id) + '.npy'))
        progbar.add(1)
    images = images.to_numpy()

    return images, image_dict


def image_feat_path_load(images, features_dir):
    """
    Get the full path for each image feature.
    """
    images = pd.Series(images)
    images = images.apply(lambda x: os.path.join(features_dir, str(x) + '.npy'))
    images = images.to_numpy()

    return images


def ans_proc(answers, label_encoder, num_classes):
    """
    Convert answers into one-hot encoding vectors.
    """
    answers = label_encoder.transform(answers)
    answers = tf.keras.utils.to_categorical(answers, num_classes)
    return answers


def get_modal_answer(answers):
    candidates = {}
    for i in range(10):
        candidates[answers[i]['answer']] = 1

    for i in range(10):
        candidates[answers[i]['answer']] += 1

    return max(candidates.items(), key=operator.itemgetter(1))[0]


def get_answer(answers, answers_type):
    """
    Get a single answer from each question. By default, each question has 10 answers.
    'mcq' extracts the machine generated answer.
    'modal' extracts the most common answer from the 10 human annotated answers.
    """
    if answers_type == "mcq":
        answers = pd.DataFrame(answers)['multiple_choice_answer']
    elif answers_type == "modal":
        answers = pd.DataFrame(answers)['answers'].apply(lambda x: get_modal_answer(x))
    else:  
        print(f'ERROR: Invalid answer type. Given {answers_type}.')
        exit(-1)

    return answers
