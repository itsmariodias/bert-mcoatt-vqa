"""
Utility to extract the coco image features and store them prior to training.
We use models pretrained on ImageNet to extract the features.
Very Deep Convolutional Networks for Large-Scale Image Recognition
Ref: https://arxiv.org/abs/1409.1556
Deep Residual Learning for Image Recognition
Ref: https://arxiv.org/abs/1512.03385

For Bottom-Up Attention, we extract the features from the pretrained files provided at
https://github.com/peteanderson80/bottom-up-attention/
Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering
Ref: https://arxiv.org/abs/1707.07998

Currently, accepted models are 'vgg19', 'resnet152', 'bottom_up_36' and 'bottom_up_100'.
"""

from configs.base_config import Config
from utils.read_tsv import read_tsv

import os
import pandas as pd
import json
import tensorflow as tf
import numpy as np


# padding with zero to get correct image filename
def pad_with_zero(num):
    total_digits = 6
    num_zeros = total_digits - len(str(num))
    return num_zeros * "0" + str(num)


# extracting and storing each extracted image feature as a .npy file.
def extract(output_dir, image_dir, image_id, image_prefix, image_postfix, preprocess_input, model):
    image_path = os.path.join(image_dir, image_prefix + pad_with_zero(str(image_id)) + image_postfix)
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(448, 448))
    img = tf.keras.preprocessing.image.img_to_array(img)

    # preprocess the image by
    # (1) expanding the dimensions to include batch dim and
    # (2) subtracting the mean RGB pixel intensity from the ImageNet dataset
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    # pass the images through the network and use the outputs as our actual features
    features = model.predict(img)  # (BATCH_SIZE, 14, 14, filter_size) (filter size: vgg19 = 512, resnet152 = 2048)
    features = tf.transpose(features, perm=[0, 3, 1, 2])  # (BATCH_SIZE, filter_size, 14, 14)
    features = tf.reshape(features, (features.shape[0], features.shape[1], -1))  # (BATCH_SIZE, filter_size, 196)
    features = tf.transpose(features, perm=[0, 2, 1])  # (BATCH_SIZE, 196, filter_size)

    with open(os.path.join(output_dir, f'{image_id}.npy'), 'wb') as f:
        np.save(f, features)


def get_image_ids(questions_path):
    # Load the image_ids from dataset
    data = json.load(open(questions_path, 'r'))['questions']

    data = pd.DataFrame(data)

    images = data['image_id']

    unique_images = set(images)

    return unique_images


def imagenet_extract(C, num_images, unique_images_list, data_list):
    # load necessary models based on model type
    if C.FEATURES_TYPE == "vgg19":
        model = tf.keras.applications.VGG19(include_top=False, weights="imagenet", input_shape=(448, 448, 3))
        preprocess_input = tf.keras.applications.vgg19.preprocess_input
    elif C.FEATURES_TYPE == "resnet152":
        model = tf.keras.applications.ResNet152(include_top=False, weights="imagenet", input_shape=(448, 448, 3))
        preprocess_input = tf.keras.applications.resnet.preprocess_input
    else:
        print(f"ERROR: Invalid coco feature type. Given {C.FEATURES_TYPE}.")
        exit(-1)

    # initialize progress bar to track
    progbar = tf.keras.utils.Progbar(num_images, verbose=1)

    # get directory of the image files
    image_dir_list = [os.path.join(C.TRAIN_DIR, "train2014"), os.path.join(C.VAL_DIR, "val2014"),
                      os.path.join(C.TEST_DIR, "test2015")]

    # prefix for each image
    prefix_list = ['COCO_train2014_000000', 'COCO_val2014_000000', 'COCO_test2015_000000']

    for i in range(len(data_list)):
        data, unique_images, image_dir, prefix = data_list[i], unique_images_list[i], image_dir_list[i], prefix_list[i]

        if not os.path.exists(image_dir):
            print(f'ERROR: {image_dir} does not exist. Please store COCO {data} images here.')
            exit(-1)

        # begin extraction process
        for image_id in unique_images:
            extract(C.FEATURES_DIR, image_dir, image_id, prefix, ".jpg", preprocess_input, model)
            progbar.add(1)


def bottom_up_extract(C, num_images, data_list):
    # initialize progress bar to track
    progbar = tf.keras.utils.Progbar(num_images, verbose=1)

    # get directory of the tsv files
    if C.IMG_SEQ_LEN == 100:
        trainval_file = os.path.join(C.DATA_DIR, "tsv", "trainval")
        test_file = os.path.join(C.DATA_DIR, "tsv", "test2015")
    elif C.IMG_SEQ_LEN == 36:
        trainval_file = os.path.join(C.DATA_DIR, "tsv", "trainval_36")
        test_file = os.path.join(C.DATA_DIR, "tsv", "test2015_36")
    else:
        print(f'ERROR: Wrong feature dimensions. Acceptable dimensions are 100 (adaptive) or 36. '
              f'Given {C.IMG_SEQ_LEN}.')
        exit(-1)

    # verify if the relevant tsv files are present in directory
    if 'train' in data_list or 'val' in data_list:
        if not os.path.exists(trainval_file):
            print(f'ERROR: {trainval_file} does not exist. Please store tsv files here.')
            exit(-1)

        for tsv_file in os.listdir(trainval_file):
            read_tsv(os.path.join(trainval_file, tsv_file), C.FEATURES_DIR, progbar)
    if 'test' in data_list:
        if not os.path.exists(test_file):
            print(f'ERROR: {test_file} does not exist. Please store tsv files here.')
            exit(-1)

        for tsv_file in os.listdir(test_file):
            read_tsv(os.path.join(test_file, tsv_file), C.FEATURES_DIR, progbar)


def coco_extract(C):
    # create directory if it doesn't exist
    os.makedirs(C.FEATURES_DIR, exist_ok=True)

    data_list = ['train', 'val']
    if (C.TRAIN_SPLIT != 'train' and C.EVAL) or C.RUN_MODE == 'test':
        data_list.append('test')

    unique_images_list = []

    # get list of unique image ids
    for data in data_list:
        unique_images_list.append(get_image_ids(C.QUESTION_PATH[data]))

    num_images = sum(len(x) for x in unique_images_list)

    if len(os.listdir(C.FEATURES_DIR)) == num_images:
        if C.VERBOSE: print(f"\nAll {C.FEATURES_TYPE} features have already been extracted. Skipping...")
        return

    if C.VERBOSE: print(f"\nAll {C.FEATURES_TYPE} features have not been extracted.")
    print("Beginning extraction process...")

    if C.FEATURES_TYPE == "bottom_up_100" or C.FEATURES_TYPE == "bottom_up_36":
        bottom_up_extract(C, num_images, data_list)
    else:
        imagenet_extract(C, num_images, unique_images_list, data_list)

    print(f"All {C.FEATURES_TYPE} features have been successfully extracted.")


if __name__ == "__main__":
    C = Config()
    coco_extract(C)
