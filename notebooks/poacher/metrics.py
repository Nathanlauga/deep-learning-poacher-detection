
from poacher import config
import tensorflow as tf
from tensorflow.keras import backend as K


# Classified as 1 if : an object is detected and the max prob is to the 'person' class
# Classified as 0 if : no object is detected or the max prob is not the 'person' class

def poacher_accuracy(y_true, y_pred):
    """
    """
    obj_conf_true = y_true[...,4]
    
    obj_conf_pred = y_pred[...,4]
    best_pred_idx = tf.argmax(y_pred[..., 5:], axis=-1)
    
    # Step 1 : Convert probability to 1 or 0
    pers_pred = tf.where((obj_conf_pred > config.THRESHOLD) & (best_pred_idx == 0), 1., 0.)
    
    # Step 2 : Get maximum pred for each frames
    pers_pred_max = tf.math.reduce_max(tf.math.reduce_max(tf.math.reduce_max(pers_pred, axis=-1), axis=-1), axis=-1)
    pers_true_max = tf.math.reduce_max(tf.math.reduce_max(tf.math.reduce_max(obj_conf_true, axis=-1), axis=-1), axis=-1)
    
    # Step 3 : Compare number of equal prediction VS True
    correct_prediction = tf.equal(pers_pred_max, pers_true_max)
    
    # Step 4 : Convert boolean to float
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    
    # Step 5 : Get accuracy 
    accuracy = tf.reduce_mean(correct_prediction)
    
    return accuracy

def poacher_confusion_matrix(y_true, y_pred):
    
    obj_conf_true = y_true[...,4]
    
    obj_conf_pred = y_pred[...,4]
    best_pred_idx = tf.argmax(y_pred[..., 5:], axis=-1)
    
    # Step 1 : Convert probability to 1 or 0
    pers_pred = tf.where((obj_conf_pred > config.THRESHOLD) & (best_pred_idx == 0), 1., 0.)
    
    # Step 2 : Get maximum pred for each frames
    pers_pred_max = tf.math.reduce_max(tf.math.reduce_max(tf.math.reduce_max(pers_pred, axis=-1), axis=-1), axis=-1)
    pers_true_max = tf.math.reduce_max(tf.math.reduce_max(tf.math.reduce_max(obj_conf_true, axis=-1), axis=-1), axis=-1)
    
    TN = tf.where((tf.equal(pers_true_max, 0.)) & (tf.equal(pers_pred_max, 0.)), 1., 0.)
    FN = tf.where((tf.equal(pers_true_max, 1.)) & (tf.equal(pers_pred_max, 0.)), 1., 0.)
    FP = tf.where((tf.equal(pers_true_max, 0.)) & (tf.equal(pers_pred_max, 1.)), 1., 0.)
    TP = tf.where((tf.equal(pers_true_max, 1.)) & (tf.equal(pers_pred_max, 1.)), 1., 0.)

    return {
        'TP': K.sum(TP), 
        'FP': K.sum(FP), 
        'FN': K.sum(FN), 
        'TN': K.sum(TN)
    }

def TP(y_true, y_pred):
    return poacher_confusion_matrix(y_true, y_pred)['TP']

def FP(y_true, y_pred):
    return poacher_confusion_matrix(y_true, y_pred)['FP']

def FN(y_true, y_pred):
    return poacher_confusion_matrix(y_true, y_pred)['FN']

def TN(y_true, y_pred):
    return poacher_confusion_matrix(y_true, y_pred)['TN']

def poacher_recall(y_true, y_pred):
    
    confusion_matrix = poacher_confusion_matrix(y_true, y_pred)
    TP = confusion_matrix['TP']
    FN = confusion_matrix['FN']
    
    return tf.divide(TP, tf.add(TP, FN))

def poacher_precision(y_true, y_pred):
    
    confusion_matrix = poacher_confusion_matrix(y_true, y_pred)
    TP = confusion_matrix['TP']
    FP = confusion_matrix['FP']
    
    return tf.divide(TP, tf.add(TP, FP))