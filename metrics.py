import os
import cv2
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skimage
import glob
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

import disc
from pandas.core.common import flatten

# Root directory of the project
ROOT_DIR = os.getcwd()

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

custom_WEIGHTS_PATH = sorted(glob.glob("logs/*/mask_rcnn_*.h5"))[-1]


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

config = disc.DiscConfig()
custom_DIR = os.path.join(ROOT_DIR, "dataset")

class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Device to load the neural network on.
# Useful if you're training a model on the same 
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"

dataset = disc.DiscDataset()
dataset.load_disc(custom_DIR, "test")

# Must call before using the dataset
dataset.prepare()

print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)

# load the last model you trained
# weights_path = model.find_last()[1]

# Load weights
print("Loading weights ", custom_WEIGHTS_PATH)
model.load_weights(custom_WEIGHTS_PATH, by_name=True)


def get_ax(rows=1, cols=1, size=16):
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


def compute_batch_ap(image_ids):
    ap_total = []
    precision_total = []
    recall_total = []
    overlap_total = []
    f1_total = []
    for image_id in image_ids:
        # Load image
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset, config,
                                   image_id, use_mini_mask=False)
        # Run object detection
        results = model.detect([image], verbose=0)
        # Compute AP
        r = results[0]
        AP, precisions, recalls, overlaps =\
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask, r['rois'], r['class_ids'], r['scores'], r['masks'], 0.001)
      
        overlaps_res = [np.max(i) for i in overlaps]
        precisions = np.mean(precisions)
        recalls = np.mean(recalls)
        #ax = get_ax(1)
        #visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
        #                      dataset.class_names, r['scores'], ax=ax,
        #                      title="Predictions")


        f1 = 2*((precisions*recalls)/(precisions+recalls))
        
        ap_total.append(AP)
        precision_total.append(precisions)
        recall_total.append(recalls)
        overlap_total.append(overlaps_res)
        f1_total.append(f1)

        info = dataset.image_info[image_id]
        '''
        print('------------------------')
        print('F1', f1)
        print('Image Id:', info["id"])
        print('Image path:', dataset.image_reference(image_id))
        print('AP: %.3f'%AP)
        print('Prec: %.3f'%np.mean(precisions))
        print('Recall: %.3f'%np.mean(recalls))
        print(precisions)
        print(recalls)
        print(f1)
        print('mean IoU: %.8f'%np.mean( overlaps))
        print('min IoU: %.3f'%np.min(overlaps))
        print('max IoU: %.3f'%np.max(overlaps))
        print('------------------------')
        '''
    return ap_total, precision_total, recall_total, overlap_total, f1_total


image_ids = dataset.image_ids
#image_ids = np.random.choice(dataset.image_ids, 1)
#image_ids = [3281]
print('IDS', image_ids)
ap_total, precision_total, recall_total, overlap_total, f1_total = compute_batch_ap(image_ids)

precision_flat = list(flatten(precision_total))
recall_flat = list(flatten(recall_total)) 
overlap_flat = list(flatten(overlap_total))
f1_flat = list(flatten(f1_total))

print('-------------------------------')
print('mAP @ IoU=50: %.3f'%np.mean(ap_total))
print('mean Precision: %.3f'%np.mean(precision_flat))
print('mean Recall: %.3f'%np.mean(recall_flat))
print('- - - - - - - - - - - - - - - -')
print('min F1: %.3f'%np.min(f1_flat))
print('mean F1: %.3f'%np.mean(f1_flat))
print('max F1: %.3f'%np.max(f1_flat))
print('- - - - - - - - - - - - - - - -')
print('min IoU: %.3f'%np.min(overlap_flat))
print('mean IoU: %.3f'%np.mean(overlap_flat))
print('max IoU: %.3f'%np.max(overlap_flat))
print('-------------------------------')
