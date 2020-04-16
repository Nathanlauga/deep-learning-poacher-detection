import pandas as pd
import numpy as np
import gc
import cv2
import tensorflow as tf

import core.config as config

def load_labels(path):
    """ Load labels from path, removes useless boxes and keep only usefull columns

    Parameters
    ----------
    path: str
        path to the csv with labels

    Returns
    -------
    pd.DataFrame:
        labels formated
    """
    labels = pd.read_csv(path)
    labels = labels[labels['lost'] != 1]
    labels = labels[labels['occluded'] != 1]
    labels = labels[['x_min','y_min','x_max','y_max','frame','video']]
    labels['class'] = 0
    return labels.drop_duplicates()


def get_bboxes_from_path(img_path, labels):
    """
    """
    video, frame = img_path.split('/')[-1].split('_')
    frame = frame[:-4]
    
    bboxes = labels[(labels['video'] == video) & (
                    labels['frame'] == int(frame))]
    bboxes = bboxes[['x_min','y_min','x_max','y_max','class']].values
    
    return bboxes if len(bboxes) > 0 else None

def image_preprocess(image, target_size, gt_boxes=None, scale=False):
    """
    """
    ih, iw    = target_size
    h,  w, _  = image.shape

    # Scale real image size x3
    if scale:
        h, w = h*3, w*3

    scale = min(iw/w, ih/h)
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_paded = image_paded / 255.

    if gt_boxes is None:
        return image_paded, None

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes
    
def bbox_iou(boxes1, boxes2):

        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)

        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                                boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                                boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area

        return inter_area / union_area
    

def preprocess_true_boxes(bboxes):
    """Format bboxes formated as (x_min, y_min, x_max, y_max, class) to 
    true bboxes shaped (grid_size, grid_size, nb_anchors, 5 + num_class).
    """    
    label = [np.zeros((config.OUT_SIZE[i], config.OUT_SIZE[i], config.ANCHOR_PER_SCALE,
                       5 + config.NUM_CLASS)) for i in range(3)]
    bboxes_xywh = [np.zeros((config.MAX_BBOX_PER_SCALE, 4)) for _ in range(3)]
    bbox_count = np.zeros((3,))

    if bboxes is not None:
        for bbox in bboxes:
            bbox_coor = bbox[:4]
            bbox_class_ind = bbox[4]
            
            onehot = np.zeros(config.NUM_CLASS, dtype=np.float)
            onehot[bbox_class_ind] = 1.0
            uniform_distribution = np.full(config.NUM_CLASS, 1.0 / config.NUM_CLASS)
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution
            
            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)
            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / config.STRIDES[:, np.newaxis]

            iou = []
            exist_positive = False
            for i in range(3):
                anchors_xywh = np.zeros((config.ANCHOR_PER_SCALE, 4))
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                anchors_xywh[:, 2:4] = config.ANCHORS[i]

                iou_scale = bbox_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)

                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:] = smooth_onehot

                    bbox_ind = int(bbox_count[i] % config.MAX_BBOX_PER_SCALE)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / config.ANCHOR_PER_SCALE)
                best_anchor = int(best_anchor_ind % config.ANCHOR_PER_SCALE)
                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh 
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot
                
                bbox_ind = int(bbox_count[best_detect] % config.MAX_BBOX_PER_SCALE)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1

    label_sbbox, label_mbbox, label_lbbox = label

    batch_smaller_target = label_sbbox 
    batch_medium_target  = label_mbbox
    batch_larger_target  = label_lbbox

    return batch_smaller_target, batch_medium_target, batch_larger_target


def get_one_img_true_bboxes(img_path, labels):
    """Retrieves true bboxes (same as the output of Yolov3 model) for 
    a given image path.
    """    
    # Step 1 : get the x_min, y_min, x_max, y_max, class of the image
    bboxes = get_bboxes_from_path(img_path, labels)
    
    # Step 2 : read the image
    image = cv2.imread(img_path)
    
    # Step 3 : preprocess the image so that it can be use for the model
    image, bboxes = image_preprocess(np.copy(image), [config.INPUT_SIZE, config.INPUT_SIZE], bboxes, scale=True)
    
    # Step 4 : retrieves true bboxes formated same as the model output.
    smaller_target, medium_target, larger_target = preprocess_true_boxes(bboxes)
    
    # Clean memory
    gc.collect()

    return image, (smaller_target, medium_target, larger_target)


def draw_bbox(image, bboxes, show_label=True):
    """
    bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
    """

    image_h, image_w, _ = image.shape
    colors = [(255,255,255)]

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        fontScale = 0.5
        score = bbox[4]
        bbox_color = colors[0]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        if show_label:
            bbox_mess = '%s: %.2f' % ('person', score)
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick//2)[0]
            cv2.rectangle(image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3), bbox_color, -1)  # filled

            cv2.putText(image, bbox_mess, (c1[0], c1[1]-2), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick//2, lineType=cv2.LINE_AA)

    return image

def bboxes_iou(boxes1, boxes2):

    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up       = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down    = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area    = inter_section[..., 0] * inter_section[..., 1]
    union_area    = boxes1_area + boxes2_area - inter_area
    ious          = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious

def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    """
    :param bboxes: (xmin, ymin, xmax, ymax, score, class)
    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]

        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)

            assert method in ['nms', 'soft-nms']

            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0

            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes

def postprocess_boxes(pred_bbox, org_img_shape, input_size, score_threshold):

    valid_scale=[0, np.inf]
    pred_bbox = np.array(pred_bbox)

    pred_xywh = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]

    # # (1) (x, y, w, h) --> (xmin, ymin, xmax, ymax)
    pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
    # # (2) (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
    org_h, org_w = org_img_shape
    resize_ratio = min(input_size / org_w, input_size / org_h)

    dw = (input_size - resize_ratio * org_w) / 2
    dh = (input_size - resize_ratio * org_h) / 2

    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

    # # (3) clip some boxes those are out of range
    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0

    # # (4) discard some invalid boxes
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

    # # (5) discard some boxes with low scores
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores > score_threshold
    mask = np.logical_and(scale_mask, score_mask)
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

    return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)


def frame_with_true_bboxes(img_path, labels):
    """
    """
    frame = cv2.imread(img_path)
    frame_size = frame.shape[:2]
    image_data, image_target = get_one_img_true_bboxes(img_path, labels)
    
    image_data = image_data[np.newaxis, ...].astype(np.float32)
    pred_bbox = list()
    for l in range(3):
        pred_bbox.append(image_target[l][np.newaxis, ...])
    
    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)
    
    bboxes = postprocess_boxes(pred_bbox, frame_size, 416, 0.3)
    bboxes = nms(bboxes, 0.45, method='nms')
    
    image = draw_bbox(frame, bboxes)
    
    result = np.asarray(image)
    result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    del bboxes, image
    gc.collect()
    
    return result

def image_preporcess(image, target_size, gt_boxes=None):

    ih, iw    = target_size
    h,  w, _  = image.shape

    scale = min(iw/w, ih/h)
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_paded = image_paded / 255.

    if gt_boxes is None:
        return image_paded

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes

def predict_on_frame(frame, model, input_size=416, threshold=0.3, verbose=None):
    """
    """
    # Preprocess image (change size and convert to np array)
    frame_size = frame.shape[:2]
    image_data = image_preporcess(np.copy(frame), [input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)
    
    gc.collect()

    # Predict using keras model
    pred_bbox = model.predict_on_batch(image_data)

    del image_data
    gc.collect()

    # Post process boxes to so that they can be draw on image
    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)
    if verbose is not None:
        print('Boxes with object detected inside')
        print(np.round(np.array([p for p in pred_bbox[:, 0:5] if p[...,4] > threshold])))
    
    bboxes = postprocess_boxes(pred_bbox, frame_size, input_size, threshold)
    
    # Keep only person object prediction
    bboxes = bboxes[bboxes[..., 5] == 0]

    bboxes = nms(bboxes, 0.45, method='nms')
    print('Num person detected:', len(bboxes))

    if verbose is not None:
        print('Boxes with persons detected inside')
        print(bboxes)

    # Draw box on image
    image = draw_bbox(frame, bboxes)
    
    del frame, pred_bbox
    gc.collect()

    # Convert image to np array
    result = np.asarray(image)
    
    # Add an information at the top of the image
    info = 'no object' if len(bboxes) < 1 else 'object detected'
    cv2.putText(result, text=info, org=(50, 70), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=2, color=(255, 255, 255), thickness=5)
    
    result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    del bboxes, image
    gc.collect()
    
    if cv2.waitKey(1) & 0xFF == ord('q'): return None
    return result