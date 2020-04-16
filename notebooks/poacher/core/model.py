# ============================================================================================================= #
# Code imported from https://github.com/YunYang1994/TensorFlow2.0-Examples/tree/master/4-Object_Detection/YOLOV3
# ============================================================================================================= #

import core.common as common
import core.backbone as backbone
import core.config as config
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
            conv_layer.trainable = False
            bn_layer.trainable = False
        else:
            conv_layer.set_weights([conv_weights, conv_bias])
            
            # Only train on last layer because our dataset is 
            # mainly with small target.
            if i != 74:
                conv_layer.trainable = False
                
    wf.close()


def box_iou(b1, b2):
    """Return iou tensor
    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh
    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)
    """

    # Expand dim to apply broadcasting.
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.0
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.0
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.0)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou



def yolo_loss(y_true, y_pred, output_layer=0):
    '''Return yolo_loss tensor
    Parameters
    ----------
    yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
    y_true: list of array, the output of preprocess_true_boxes
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    ignore_thresh: float, the iou threshold whether to ignore object confidence loss
    Returns
    -------
    loss: tensor, shape=(1,)
    '''
    anchors = config.ANCHORS
    num_classes = config.NUM_CLASS
    ignore_thresh = 0.5
    print_loss = False
    
    num_layers = len(anchors)//3 
    yolo_outputs = y_pred
    
    anchor_mask = (
        [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]
    )
    # anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]
    input_shape = K.cast(K.shape(yolo_outputs)[1:3] * 32, K.dtype(y_true[0]))
    grid_shapes = K.cast(K.shape(yolo_outputs)[1:3], K.dtype(y_true[0]))
    
    loss = 0
    m = K.shape(yolo_outputs)[0] # batch size, tensor
    mf = K.cast(m, K.dtype(yolo_outputs))
    
    l = output_layer

    object_mask = y_true[..., 4:5]
    true_class_probs = y_true[..., 5:]

    # Preprocess predictions for the loss
    raw_pred = y_pred
    
    pred_xy = y_pred[..., :2]
    pred_wh = K.log(
        y_pred[..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1]
    )
    pred_box = K.concatenate([pred_xy, pred_wh])

    # Darknet raw box to calculate loss.
    raw_true_xy = y_true[..., :2] # *grid_shapes[::-1] - grid
    raw_true_wh = K.log(
        y_true[..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1]
    )
    raw_true_wh = K.switch(
        object_mask, raw_true_wh, K.zeros_like(raw_true_wh)
    )  # avoid log(0)=-inf
    
    # Find ignore mask, iterate over each of batch.
    ignore_mask = tf.TensorArray(K.dtype(y_true), size=1, dynamic_size=True)
    object_mask_bool = K.cast(object_mask, 'bool')

    def loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(
                y_true[b, ..., 0:4], object_mask_bool[b, ..., 0]
            )
            iou = box_iou(pred_box[b], true_box)
            best_iou = K.max(iou, axis=-1)
            ignore_mask = ignore_mask.write(
                b, K.cast(best_iou < ignore_thresh, K.dtype(true_box))
            )
            return b + 1, ignore_mask

    _, ignore_mask = tf.while_loop(
            lambda b, *args: b < m, loop_body, [0, ignore_mask]
    )
    
    ignore_mask = ignore_mask.stack()
    ignore_mask = K.expand_dims(ignore_mask, -1)
    
    # K.binary_crossentropy is helpful to avoid exp overflow.
    xy_loss = (
        object_mask
        * K.binary_crossentropy(tf.sigmoid(raw_true_xy), tf.sigmoid(y_pred[..., 0:2]), from_logits=True)
    )
    wh_loss = (
        object_mask
        * 0.5
        * K.square(raw_true_wh - y_pred[..., 2:4])
    )
    confidence_loss = (
        object_mask
        * K.binary_crossentropy(object_mask, y_pred[..., 4:5], from_logits=True)
        + (1 - object_mask)
        * K.binary_crossentropy(object_mask, y_pred[..., 4:5], from_logits=True)
        * ignore_mask
    )
    class_loss = object_mask * K.binary_crossentropy(
        true_class_probs, y_pred[..., 5:], from_logits=True
    )
    
    xy_loss = K.sum(xy_loss) / mf
    wh_loss = K.sum(wh_loss) / mf
    confidence_loss = K.sum(confidence_loss) / mf
    class_loss = K.sum(class_loss) / mf
    loss += xy_loss + wh_loss + confidence_loss + class_loss
    
    if print_loss:        
        tf.print('For output layer %i loss, xy_loss, wh_loss, confidence_loss, class_loss is :'%(l))
        tf.print(loss)
        tf.print(xy_loss)
        tf.print(wh_loss)
        tf.print(confidence_loss)
        tf.print(class_loss)
        
    return loss