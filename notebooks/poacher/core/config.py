
import numpy as np

def get_anchors(anchors):
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)

NUM_CLASS = 80
ANCHORS = get_anchors('1.25,1.625, 2.0,3.75, 4.125,2.875, 1.875,3.8125, 3.875,2.8125, 3.6875,7.4375, 3.625,2.8125, 4.875,6.1875, 11.65625,10.1875')
# ANCHORS = get_anchors('10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326')
STRIDES = np.array([8,16,32])
IOU_LOSS_THRESH = 0.3

INPUT_SIZE = 416
OUT_SIZE = [52, 26, 13]
ANCHOR_PER_SCALE = 3
MAX_BBOX_PER_SCALE = 150