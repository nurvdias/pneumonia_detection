import os
import sys

sys.path.append('Mask_RCNN')
sys.path.append('pneumonia_detection')

import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
from mrcnn import utils

import random
import math
import numpy as np
import cv2
import json
import pydicom
from imgaug import augmenters as iaa
from tqdm import tqdm
import pandas as pd
import glob
from keras.callbacks import CSVLogger

import utils.utils as ut

from pneumonia_detection_config import PneumoniaDetectionConfig as PDConfig
from pneumonia_detection_dataset_2 import PneumoniaDetectionDataset as PDDataset
from pneumonia_detection_config import PneumoniaDetectionInferenceConfig as PDInferenceConfig

DATASET_DIR = 'dataset'
TRAIN_IMAGES = DATASET_DIR + os.sep + 'train_images'
TEST_IMAGES = DATASET_DIR + os.sep + 'test_images'
MODEL_DIR = 'logs'
MODEL = os.path.join(MODEL_DIR, "mask_rcnn_pneumonia_detection.h5")
ANALYSIS_DIR = 'analysis'

ORIG_DICOM_SIZE = 1024

# Make logs directory if it doesn't exist.
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)


class Config128(PDConfig):
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128
    RPN_ANCHOR_SCALES = (4, 8, 16, 32, 64)
    STEPS_PER_EPOCH = 1000
    LEARNING_RATE = 0.01


class Config512(PDConfig):
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    STEPS_PER_EPOCH = 1000
    LEARNING_RATE = 0.01


class Config1024(PDConfig):
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    STEPS_PER_EPOCH = 1000


def split_data(data):
    """
    Split dataset into train_data, val_data, test_data

    :return: train_data, val_data, test_data
    """

    all_dataset = ut.parse_data(TRAIN_IMAGES, data)
    all_dataset = list(all_dataset.values())  # We only need the dict values

    # Split dataset into train_data and val_data in the ratio 9:1
    random.seed(42)
    random.shuffle(all_dataset)

    split = .25
    split_idx = int((1 - split) * len(all_dataset))

    train_data = all_dataset[:split_idx]
    val_test_data = all_dataset[split_idx:]

    val_split = .8
    idx = int((1 - val_split) * len(val_test_data))

    val_data = val_test_data[idx:]
    test_data = val_test_data[:idx]

    return train_data, val_data, test_data


def prepare_dataset(data):
    """
    Load dataset in PDDateset

    :param data: dataset to load
    :return: loaded dataset
    """
    dataset = PDDataset(data, ORIG_DICOM_SIZE, ORIG_DICOM_SIZE)
    dataset.prepare()

    return dataset


def load_training_weights(model, config, init_training=False):
    """
    Load training weights by determining if there is already weights for
    pneumonia a detection model. If there's none, use coco weights.
    If initial training is set to false, then continue training by loading
    last trained epoch weights

    @params: model -> pneumonia detection model
    @params: config -> pneumonia detection configuration
    @params: init_training -> boolean indicating if training is conitnuing
                or starting afresh
    @return: training weights path
    """

    weights_path = ''
    pre_trained = False

    if os.path.exists(MODEL) and init_training:
        # If model weights exist and training is intended to start
        # from epoch 1
        print('Loading pneumonia detection model.')
        weights_path = MODEL
        pre_trained = False
        print('Pneumonia detection trained model found in {}'.format(weights_path))
    elif not os.path.exists(MODEL) and init_training:
        # Download and use COCO dataset weights
        # if model path does not exist and training is intended to start
        # from epoch 1
        weights_path, pre_trained = load_coco_weights(model)
    elif not init_training:
        # If continuing training.
        # Try loading last trained model and continue training
        # else load coco dataset model and start training
        # from epoch 1 -> fall back
        try:
            print('Loading last trained weights.')
            weights_path = ut.load_last_model(model.model_dir, config)
            model.load_weights(weights_path, by_name=True)
            pre_trained = True
            print('Pneumonia detection trained model found in {}.'.format(weights_path))
        except:
            weights_path, pre_trained = load_coco_weights(model)

    return weights_path, pre_trained


def load_coco_weights(model):
    print('Weights not found. Attempting to download pretrained COCO weights.')
    weights_path = os.path.join(MODEL_DIR, 'mask_rcnn_coco.h5')
    if not os.path.exists(weights_path):
        utils.download_trained_weights(weights_path)

    # Exclude the last layers because they require a matching
    # number of classes
    print('Loading COCO weights.')
    model.load_weights(weights_path, by_name=True, exclude=[
        "mrcnn_class_logits", "mrcnn_bbox_fc",
        "mrcnn_bbox", "mrcnn_mask"])
    pre_trained = False
    print('COCO weights loaded.')

    return weights_path, pre_trained


def load_weights_for_512(model, config, init_training=True):
    """
    Load training weights after training with dim 128 X 128
    """

    if init_training:
        weights_path = os.path.join(MODEL_DIR, 'mask_rcnn_pneumonia_detection_128.h5')
        print('Loading weights from {}'.format(weights_path))
        model.load_weights(weights_path, by_name=True)
        pre_trained = False
    else:
        weights_path = ut.load_last_model(model.model_dir, config)
        print('Continuing training weights from {}'.format(weights_path))
        model.load_weights(weights_path, by_name=True)
        pre_trained = True

    return weights_path, pre_trained


def load_weights_for_1024(model, config, init_training=True):
    """
    Load training weights after training with dim 512 X 512
    """

    if init_training:
        weights_path = os.path.join(MODEL_DIR, 'mask_rcnn_pneumonia_detection_512.h5')
        print('Loading weights from {}'.format(weights_path))
        model.load_weights(weights_path, by_name=True)
        pre_trained = False
    else:
        weights_path = ut.load_last_model(model.model_dir, config)
        print('Continuing training weights from {}'.format(weights_path))
        model.load_weights(weights_path, by_name=True)
        pre_trained = True

    return weights_path, pre_trained


def train(train_dataset, val_dataset, init_training_1=True, init_training_2=True,
          init_training_3=True):
    """
    Train the model in 3 steps using the training and validation datasets
    1st step training of 10000 iterations at image size 128x128
    2nd step training of 10000 iterations at image size 512x512
    3rd step training of 10000 iterations at image size 1024x1024

    At the end of each step, save the trained weights (the final weight is saved at the end of step
    3).

    :param train_dataset:
    :param val_dataset:
    :param init_training_1:
    :param init_training_2:
    :param init_training_3:
    :return:
    """

    # Image augmentation
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
    augmentation = iaa.SomeOf((0, 2), [
        iaa.Affine(
            scale={"x": (1.0, 2.0), "y": (1.0, 2.0)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-25, 25),
            shear=(-8, 8)
        ),
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.Multiply((0.8, 1.5))
    ])

    # 1st step training
    print('Training with image size 128x128')
    config = Config128()
    model = modellib.MaskRCNN(mode='training', config=config, model_dir=MODEL_DIR)

    if init_training_1:
        weights_path, pre_trained = load_training_weights(model, config, True)
    else:
        weights_path, pre_trained = load_training_weights(model, config, False)

    print('Weights location: {}'.format(weights_path))

    if pre_trained:
        logger = CSVLogger(os.path.join(MODEL_DIR, 'training_log_128.csv'), append=True)
    else:
        logger = CSVLogger(os.path.join(MODEL_DIR, 'training_log_128.csv'), append=False)

    model.train(train_dataset, val_dataset, learning_rate=config.LEARNING_RATE,
                epochs=10000, layers='all', augmentation=augmentation, custom_callbacks=[logger])

    model.keras_model.save_weights(os.path.join(MODEL_DIR, "mask_rcnn_pneumonia_detection_128.h5"))

    # 2nd step training
    print('Training with image size 512x512')
    config = Config512()
    model = modellib.MaskRCNN(mode='training', config=config, model_dir=MODEL_DIR)

    if init_training_2:
        weights_path, pre_trained = load_weights_for_512(model, config, True)
    else:
        weights_path, pre_trained = load_weights_for_512(model, config, False)

    print('Weights location: {}'.format(weights_path))

    if pre_trained:
        logger = CSVLogger(os.path.join(MODEL_DIR, 'training_log_512.csv'), append=True)
    else:
        logger = CSVLogger(os.path.join(MODEL_DIR, 'training_log_512.csv'), append=False)

    model.train(train_dataset, val_dataset, learning_rate=config.LEARNING_RATE,
                epochs=10000, layers='all', augmentation=augmentation, custom_callbacks=[logger])

    model.keras_model.save_weights(os.path.join(MODEL_DIR, "mask_rcnn_pneumonia_detection_512.h5"))

    # 3rd step training
    print('Training with image size 1024x1024')
    config = Config1024()
    model = modellib.MaskRCNN(mode='training', config=config, model_dir=MODEL_DIR)

    if init_training_3:
        weights_path, pre_trained = load_weights_for_1024(model, config, True)
    else:
        weights_path, pre_trained = load_weights_for_1024(model, config, False)

    print('Weights location: {}'.format(weights_path))

    if pre_trained:
        logger = CSVLogger(os.path.join(MODEL_DIR, 'training_log_1024.csv'), append=True)
    else:
        logger = CSVLogger(os.path.join(MODEL_DIR, 'training_log_1024.csv'), append=False)

    model.train(train_dataset, val_dataset, learning_rate=config.LEARNING_RATE,
                epochs=10000, layers='all', augmentation=augmentation, custom_callbacks=[logger])

    model.keras_model.save_weights(MODEL)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Mask R-CNN to detect pneumonia.')
    parser.add_argument('--first_step', required=False, metavar='True/False',
                        help='Train first step afresh or continue from previous training')
    parser.add_argument('--second_step', required=False, metavar='True/False',
                        help='Train second step afresh or continue from previous training')
    parser.add_argument('--third_step', required=False, metavar='True/False',
                        help='Train third step afresh or continue from previous training')
    args = parser.parse_args()

    first_step = True
    second_step = True
    third_step = True

    if args.first_step is not None:
        first_step = bool(args.first_step)
    if args.second_step is not None:
        second_step = bool(args.second_step)
    if args.third_step is not None:
        third_step = bool(args.third_step)

    df = pd.read_csv(os.path.join(DATASET_DIR, 'train_labels.csv'))
    train_data, val_data, test_data = split_data(df)

    train_dataset = prepare_dataset(train_data)
    val_dataset = prepare_dataset(val_data)
    test_dataset = prepare_dataset(test_data)

    train(train_dataset, val_dataset, first_step, second_step, third_step)

