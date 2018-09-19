import numpy as np
import keras.backend as K



def validate_matrices(kwargs):
    """

    :param kwargs: kwargs from machine learning modules containing data matrices (X/y) or (X_train/X_test/y_train/y_test)
    :return: True if type is correct and does not contain NAs
    """
    assert kwargs is not None, 'Class Shallow Model must be initialized with at least 2 arguments.'
    if len(kwargs.keys()) <= 3:
        assert np.isfinite(kwargs['X']).all(), 'Matrix X with NAs or infinite values.'
        assert np.isfinite(kwargs['y']).all(), 'Matrix y with NAs or infinite values.'
        assert type(kwargs['X']) == np.ndarray, 'Matrix X must be of type numpy.ndarray.'
        assert type(kwargs['y']) == np.ndarray, 'Matrix y must be of type numpy.ndarray.'
    elif len(kwargs.keys()) >= 4:
        assert np.isfinite(kwargs['X_train']).all(), 'Matrix X_train with NAs or infinite values.'
        assert np.isfinite(kwargs['X_test']).all(), 'Matrix X_test with NAs or infinite values.'
        assert np.isfinite(kwargs['y_train']).all(), 'Matrix y_train with NAs or infinite values.'
        assert np.isfinite(kwargs['y_test']).all(), 'Matrix y_test with NAs or infinite values.'
        assert type(kwargs['X_train']) == np.ndarray, 'Matrix X_train must be of type numpy.ndarray.'
        assert type(kwargs['X_test']) == np.ndarray, 'Matrix X_test must be of type numpy.ndarray.'
        assert type(kwargs['y_train']) == np.ndarray, 'Matrix y_train must be of type numpy.ndarray.'
        assert type(kwargs['y_test']) == np.ndarray, 'Matrix y_test must be of type numpy.ndarray.'
    return True


def timer(raw_time):
    hours, rem = divmod(raw_time, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)


# External score functions for keras

def matthews_correlation(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)
    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)
    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return numerator / (denominator + K.epsilon())


def precision_k(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision



def recall_k(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def f1_score_k(y_true, y_pred):
    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))
    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0: return 0
    # How many selected items are relevant?
    precision = c1 / c2
    # How many relevant items are selected?
    recall = c1 / c3
    # Calculate f1_score
    f1_score = 2 * (precision * recall) / ((precision + recall) + K.epsilon())
    return f1_score