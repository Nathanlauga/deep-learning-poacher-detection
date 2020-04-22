from time import time
import argparse

from poacher.model import (yolo_loss)
from poacher import utils

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

import os
import gc

INPUT_SIZE = 416 
THRESHOLD = 0.3

# ======= INIT SCRIPT ====== #

print('tensorflow version:', tf.__version__)

physical_devices = tf.config.experimental.list_physical_devices('GPU')

# If use tensorflow GPU
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.per_process_gpu_memory_fraction = 0.3


# ====== LOAD MODEL ======= # 

def load_model(model_path):

    t0 = time()
    model = tf.keras.models.load_model(model_path, compile=False)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    model.compile(optimizer=optimizer, 
                    loss={
                        'tf_op_layer_concat_4': lambda y_true, y_pred: yolo_loss(y_true, y_pred), # 52x52
                        'tf_op_layer_concat_7': lambda y_true, y_pred: yolo_loss(y_true, y_pred), # 26x26
                        'tf_op_layer_concat_10': lambda y_true, y_pred: yolo_loss(y_true, y_pred),# 13x13
                    }
                    )
    print('model loaded in %.2fs'%(time() - t0))
    return model

# model_path = 'save/model_trained.h5'
model_path = 'save/20200417_model_trained.h5'
model = load_model(model_path)
gc.collect()

# ========================== #


def get_next_frame(vid):
    """
    """
    # Get next frame
    return_value, frame = vid.read()

    # If there is a frame then change color scale
    if return_value:
    #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    else:
        return None
    
def image_preprocess(image, target_size):

    ih, iw    = target_size
    h,  w, _  = image.shape

    scale = min(iw/w, ih/h)
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_paded = image_paded / 255.

    return image_paded

def get_image_preprocess(frame):

    input_size = INPUT_SIZE
    # Preprocess image (change size and convert to np array)
    frame_size = frame.shape[:2]
    image_data = image_preprocess(np.copy(frame), [input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    return image_data

def predict(X, model):
    t0 = time()
    pred = model.predict_on_batch(X)
    return pred, time() - t0

def postprocess_pred_bbox(pred_bbox, frame_size, input_size=416, threshold=0.3):

    input_size = INPUT_SIZE
    threshold = THRESHOLD

    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)

    bboxes = utils.postprocess_boxes(pred_bbox, frame_size, input_size, threshold)

    # Keep only person object prediction
    bboxes = bboxes[bboxes[..., 5] == 0]

    bboxes = utils.nms(bboxes, 0.45, method='nms')    
    return bboxes


def main():
    # Parse arguments from cmd
    parser = argparse.ArgumentParser()
    parser.add_argument("video-path", help = "Path to the video file")
    parser.add_argument("--out-path", help = "Path where to store the output video", default = "./out.mp4")
    parser.add_argument("--break-at", help = "Number of frames to retrieve in the output", default = None)
    parser.add_argument("--ignore-frames", help = "Number of frames to skip betweeb each detection", default = 0, type = int)
    parser.add_argument("--verbose", help="Whether you want details in cli or not", default = None, type = int)
    args = parser.parse_args()

    video_path = agrs['video-path']
    out_path = agrs['out-path']
    break_at = agrs['break-at']
    frames_to_ignore = agrs['ignore-frames']
    verbose = args['verbose']

    t_start = time()
    vid = cv2.VideoCapture(video_path)
    fps = round(vid.get(cv2.CAP_PROP_FPS),0)

    times = {
        'next frame': list(),
        'write time': list(),
        'preprocess time': list(),
        'pred time': list(),
        'postprocess time': list(),
        'draw bboxes time': list()
    }
    frames_analysed = 0

    cnt = 0
    # get first frame
    frame = get_next_frame(vid)

    # video resolution
    height, width = frame.shape[:2]

    frames_hist = list()

    print('Original video: %ix%i and %i fps'%(width, height, fps))

    out = cv2.VideoWriter(out_path, 
                            cv2.VideoWriter_fourcc(*'DIVX'), 
                            fps, (width, height))

    print('Ready to analyse video at %.2fs'%(time()-t_start))
    pred_bbox = []

    input_size = 416
    threshold = 0.3

    while frame is not None:
        
        if cnt%(frames_to_ignore+1) == 0:
            # Preprocess frame
            t0 = time()
            image_data = get_image_preprocess(frame)
            times['preprocess time'].append(time() - t0)

            # Predict
            pred_bbox, t = predict(image_data, model)
            times['pred time'].append(t)
            
            # Postprocess to know if there is any poacher on the image
            t0 = time()
            pred_bbox = postprocess_pred_bbox(pred_bbox, [height, width],
                                                input_size, threshold)
            times['postprocess time'].append(time() - t0)
            
            if verbose is not None:
                print('frame n°%i : %i poacher(s) detected'%(cnt, len(pred_bbox)))
            
            frames_analysed += 1
            gc.collect()
            
        
        # Draw box on image
        t0 = time()
        frame = utils.draw_bbox(frame, pred_bbox)
        times['draw bboxes time'].append(time() - t0)
        
        t0 = time()
        out.write(frame)
        times['write time'].append(time() - t0)
        
        t0 = time()
        frame = get_next_frame(vid)
        times['next frame'].append(time() - t0)
        
        cnt += 1
        if cnt%fps == 0:
            print(cnt,'frames : %.2fs'%(time()-t_start))
            gc.collect()
        
        # Break if ask
        if break_at is not None:
            if cnt >= break_at:
                break
                
    vid.release()
    out.release()

    print('==================')
    print('Script ended in %.2fs'%(time() - t_start))
    print('Average time of writing into out video (%i frames) : %.3fs (std %.4f)'%(
        cnt, np.mean(times['write time']), np.std(times['write time'])))
    print('Average time of getting next frame (%i frames) : %.3fs (std %.4f)'%(
        cnt, np.mean(times['next frame']), np.std(times['next frame'])))
    print('Average time of preprocessing a frame (%i frames) : %.3fs (std %.4f)'%(
        frames_analysed, np.mean(times['preprocess time']), np.std(times['preprocess time'])))
    print('Average time of prediction on a frame (%i frames) : %.3fs (std %.4f)'%(
        frames_analysed, np.mean(times['pred time']), np.std(times['pred time'])))
    print('Average time of postprocess bboxes (%i frames) : %.3fs (std %.4f)'%(
        frames_analysed, np.mean(times['postprocess time']), np.std(times['postprocess time'])))
    print('Average time of drawing bboxes on a frame (%i frames) : %.3fs (std %.4f)'%(
        cnt, np.mean(times['draw bboxes time']), np.std(times['draw bboxes time'])))



if __name__ == "__main__":
	try:
		main()
	except Exception as e:
		print(e)
		sys.exit(1)
    