# ============================================================================================================= #
# Code imported from https://github.com/YunYang1994/TensorFlow2.0-Examples/tree/master/4-Object_Detection/YOLOV3
# ============================================================================================================= #

import poacher.common as common
import poacher.backbone as backbone
import poacher.config as config
import numpy as np

import tensorflow as tf
from tensorflow.keras import backend as K

def YOLOv3(input_layer):
    """Create Yolov3 network architecture.
    """
    route_1, route_2, conv = backbone.darknet53(input_layer)

    conv = common.convolutional(conv, (1, 1, 1024,  512))
    conv = common.convolutional(conv, (3, 3,  512, 1024))
    conv = common.convolutional(conv, (1, 1, 1024,  512))
    conv = common.convolutional(conv, (3, 3,  512, 1024))
    conv = common.convolutional(conv, (1, 1, 1024,  512))

    conv_lobj_branch = common.convolutional(conv, (3, 3, 512, 1024))
    conv_lbbox = common.convolutional(conv_lobj_branch, (1, 1, 1024, 3*(config.NUM_CLASS + 5)), activate=False, bn=False)
    
    conv = common.convolutional(conv, (1, 1,  512,  256))
    conv = common.upsample(conv)

    conv = tf.concat([conv, route_2], axis=-1)

    conv = common.convolutional(conv, (1, 1, 768, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))

    conv_mobj_branch = common.convolutional(conv, (3, 3, 256, 512))
    conv_mbbox = common.convolutional(conv_mobj_branch, (1, 1, 512, 3*(config.NUM_CLASS + 5)), activate=False, bn=False)
    
    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.upsample(conv)

    conv = tf.concat([conv, route_1], axis=-1)

    conv = common.convolutional(conv, (1, 1, 384, 128))
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv = common.convolutional(conv, (1, 1, 256, 128))

    conv_sobj_branch = common.convolutional(conv, (3, 3, 128, 256))
    conv_sbbox = common.convolutional(conv_sobj_branch, (1, 1, 256, 3*(config.NUM_CLASS +5)), activate=False, bn=False)

    return [conv_sbbox, conv_mbbox, conv_lbbox]

def decode(conv_output, i=0):
    """
    return tensor of shape [batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes]
            contains (x, y, w, h, score, probability)
    """

    conv_shape       = tf.shape(conv_output)
    batch_size       = conv_shape[0]
    output_size      = conv_shape[1]

    conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, 3, 5 + config.NUM_CLASS))

    conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
    conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
    conv_raw_conf = conv_output[:, :, :, :, 4:5]
    conv_raw_prob = conv_output[:, :, :, :, 5: ]

    y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
    x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])

    xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
    xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, 3, 1])
    xy_grid = tf.cast(xy_grid, tf.float32)

    pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * config.STRIDES[i]
    pred_wh = (tf.exp(conv_raw_dwdh) * config.ANCHORS[i]) * config.STRIDES[i]
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)

    return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

def load_weights(model, weights_file):
    """
    I agree that this code is very ugly, but I don’t know any better way of doing it.
    """
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

    j = 0
    for i in range(75):
        conv_layer_name = 'conv2d_%d' %i if i > 0 else 'conv2d'
        bn_layer_name = 'batch_normalization_%d' %j if j > 0 else 'batch_normalization'

        conv_layer = model.get_layer(conv_layer_name)
        filters = conv_layer.filters
        k_size = conv_layer.kernel_size[0]
        in_dim = conv_layer.input_shape[-1]

        if i not in [58, 66, 74]:
            # darknet weights: [beta, gamma, mean, variance]
            bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
            # tf weights: [gamma, beta, mean, variance]
            bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
            bn_layer = model.get_layer(bn_layer_name)
            j += 1
        else:
            conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)

        # darknet shape (out_dim, in_dim, height, width)
        conv_shape = (filters, in_dim, k_size, k_size)
        conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))
        # tf shape (height, width, in_dim, out_dim)
        
        conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])
        
        if i not in [58, 66, 74]:
            conv_layer.set_weights([conv_weights])
            bn_layer.set_weights(bn_weights)
            # if i not in [57, 65, 73]:
            if i not in [73]:
                conv_layer.trainable = False
                bn_layer.trainable = False

        else:
            conv_layer.set_weights([conv_weights, conv_bias])
            
            # Only train on last layer because our dataset is 
            # mainly with small target.
            if i != 74:
                conv_layer.trainable = False
                
    wf.close()


def yolo_loss(y_true, y_pred):
    '''Return yolo_loss tensor
    
    Parameters
    ----------
    y_true: tensor, shape(m, grid_size, grid_size, 3, 85)
        the output of preprocess_true_boxes
    y_pred: tensor, shape(m, grid_size, grid_size, 3, 85)
        the prediction of an output layer
    l: int
        2 ==> 52x52 output layer
        1 ==> 26x26 output layer
        0 ==> 13x13 output layer
    
    Returns
    -------
    loss: tensor, shape=(1,)
    '''
    print_loss = False
    
    # Number of data
    m = K.shape(y_pred)[0]
    mf = K.cast(m, K.dtype(y_pred)) 
    
    y_true_xy = y_true[...,0:2]
    y_pred_xy = y_pred[...,0:2]
    y_true_wh = y_true[...,2:4]
    y_pred_wh = y_pred[...,2:4]
    y_true_conf = y_true[...,4]
    y_pred_conf = y_pred[...,4]
    
    object_mask = y_true[..., 4:5] # Contains 0. or 1. values : if 0. then there is no objects on the cell

    # Compute iou    
    intersect_wh = K.maximum(
        K.zeros_like(y_pred_wh), (y_pred_wh + y_true_wh)/2 - K.abs(y_pred_xy - y_true_xy) 
    )
    intersect_area = intersect_wh[...,0] * intersect_wh[...,1]
    true_area = y_true_wh[...,0] * y_true_wh[...,1]
    pred_area = y_pred_wh[...,0] * y_pred_wh[...,1]
    union_area = pred_area + true_area - intersect_area
    iou = intersect_area / union_area
    
    obj_loss = (
        K.binary_crossentropy(y_true_conf, y_pred_conf, from_logits=True)
    )
    
    # K.binary_crossentropy is helpful to avoid exp overflow
    xy_loss = (
        object_mask
        * K.binary_crossentropy(
            tf.sigmoid(y_true_xy), 
            tf.sigmoid(y_pred_xy), from_logits=True)
    )
    
    wh_loss = (
        object_mask
        * K.square(K.sqrt(y_true[..., 2:4]) - K.sqrt(y_pred[..., 2:4]))
    )
    
    class_loss = (
        object_mask
        * K.square(y_true[..., 5:] - y_pred[..., 5:])
    )

    confidence_loss = (
        K.square(y_true_conf*iou - y_pred_conf)*y_true_conf
    ) 
    
    obj_loss = K.sum(obj_loss) / mf
    xy_loss = K.sum(xy_loss) / mf
    wh_loss = K.sum(wh_loss) / mf
    confidence_loss = K.sum(confidence_loss) / mf
    class_loss = K.sum(class_loss) / mf
    
    loss = obj_loss + xy_loss + wh_loss + confidence_loss + class_loss
    
    if print_loss:        
        tf.print('For output layer %i loss, obj_loss, xy_loss, wh_loss, confidence_loss, class_loss is :'%(l))
        tf.print(loss)
        tf.print(obj_loss)
        tf.print(xy_loss)
        tf.print(wh_loss)
        tf.print(confidence_loss)
        tf.print(class_loss)
        
    return loss