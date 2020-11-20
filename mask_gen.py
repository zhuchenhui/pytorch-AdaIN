import torch
import torchvision
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import matplotlib.pyplot as plt


def generate_mask_dictionary(cfg, detectron2_result, img):
    num_mask = detectron2_result['instances'].pred_masks.shape[0]
    class_res = detectron2_result['instances'].pred_classes
    scores = detectron2_result['instances'].scores
    class_dictionary = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
    dictionary = {}
    bg = 255*np.ones(img.shape, np.uint8)
    for i in range(num_mask):
        class_name = class_dictionary[class_res[i].item()]
        mask_info = detectron2_result['instances'].pred_masks[i]
        score = scores[i]
        if score < 0.99:
            continue
        mask_res = np.zeros(img.shape, np.uint8)
        for x_idx in range(img.shape[0]):
            for y_idx in range(img.shape[1]):
                if mask_info[x_idx][y_idx]:
                    for c in range(3):
                        mask_res[x_idx][y_idx][c] = 255
                        bg[x_idx][y_idx][c] = 0
        if class_name not in dictionary:
            dictionary[class_name] = [mask_res]
        else:
            dictionary[class_name].append(mask_res)
    dictionary['bg'] = [bg]
    return dictionary


def show_mask(detector_res):
    for key in detector_res.keys():
        print(key)
        num_mask = len(detector_res[key])
        if num_mask == 1:
            plt.imshow(detector_res[key][0])
        else:
            f, axarr = plt.subplots(1,len(detector_res[key]))
            for i in range(len(detector_res[key])):
                axarr[i].imshow(detector_res[key][i])
        plt.show()


def load_mask_gen():
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    return cfg, predictor
