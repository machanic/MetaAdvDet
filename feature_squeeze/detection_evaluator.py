from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import sklearn
import torch
from config import IN_CHANNELS, IMAGE_SIZE
from feature_squeeze.feature_squeeze_detector import FeatureSqueezingDetector
from sklearn.metrics import accuracy_score

def get_tpr_fpr(true_labels, pred_labels):
    TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
    FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))

    AP = np.sum(true_labels)
    AN = np.sum(1-true_labels)

    tpr = TP/AP if AP>0 else np.nan
    fpr = FP/AN if AN>0 else np.nan

    return tpr, fpr, TP, AP


def evalulate_detection_test(Y_detect_test, Y_detect_pred):
    accuracy = sklearn.metrics.accuracy_score(Y_detect_test, Y_detect_pred, normalize=True, sample_weight=None)
    tpr, fpr, tp, ap = get_tpr_fpr(Y_detect_test, Y_detect_pred)
    return accuracy, tpr, fpr, tp, ap


class DetectionEvaluator:
    """
    Get a dataset;
        Failed adv as benign / Failed adv as adversarial.
    For each detector:
        Train
        Test
        Report performance
            Detection rate on each attack.
            Detection on SAEs / FAEs.
            ROC-AUC.
    A detector should have this simplified interface:
        Y_pred = detector(X)
    """
    def __init__(self, model, dataset_name):
        # set_base_model()
        self.model = model
        self.dataset_name = dataset_name
        self.detector = self.get_detector(model)


    def get_detector(self, model):
        detector = FeatureSqueezingDetector(model)
        return detector


    def evaluate_detections(self, train_imgs, val_loader):
        accuracy_list = []

        for support_images, support_labels, query_images, query_labels, positive_labels in val_loader:
            assert support_images.shape[0] == query_images.shape[0]
            positive_labels = positive_labels.detach().cpu().numpy()
            query_labels = query_labels.detach().cpu().numpy()
            query_labels = (query_labels == positive_labels).astype(np.int64)
            query_images = query_images.detach().cpu().numpy()
            support_labels = support_labels.detach().cpu().numpy()
            support_labels = (support_labels == positive_labels).astype(np.int64)
            support_images = support_images.detach().cpu().numpy()

            for task_idx in range(query_images.shape[0]):
                for idx, label_ in enumerate(query_labels[task_idx]):
                    if label_ == 1:
                        train_imgs.append(
                            query_images[task_idx][idx].reshape(IN_CHANNELS[self.dataset_name], IMAGE_SIZE[self.dataset_name][0],
                                                       IMAGE_SIZE[self.dataset_name][1]))

                self.detector.train(np.stack(train_imgs), np.ones(len(train_imgs)))
                X_test, Y_test =  query_images[task_idx], query_labels[task_idx]
                X_test = X_test.reshape(-1, IN_CHANNELS[self.dataset_name], IMAGE_SIZE[self.dataset_name][0], IMAGE_SIZE[self.dataset_name][1])
                # Example: --detection "FeatureSqueezing?distance_measure=l1&squeezers=median_smoothing_2,bit_depth_4;
                Y_test_pred, Y_test_pred_score = self.detector.test(X_test)
                # accuracy, tpr, fpr, tp, ap = evalulate_detection_test(Y_test, Y_test_pred)
                accuracy = accuracy_score(Y_test, Y_test_pred)
                accuracy_list.append(accuracy)

        accuracy = np.mean(accuracy_list)
        print(accuracy)
        return accuracy

