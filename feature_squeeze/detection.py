from feature_squeeze.squeeze import reduce_precision_torch
import torch
from feature_squeeze.utils import model_eval_distance, model_eval_distance_dual_input

class FeatureSqueezeDetection(object):
    def __init__(self, model):
        self.model = model



    def calculate_l1_distance_fgsm(self, predictions_orig_func, predictions_squeeze_func, adv_x_dict, csv_fpath):
        print("\n===Calculating L1 distance with feature squeezing...")
        eps_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        l1_dist = []
        for i, eps in enumerate(eps_list):
            print("Epsilon=", eps)
            X_test_adv = adv_x_dict[eps]
            predictions_orig = predictions_orig_func(X_test_adv)
            predictions_squeezed = predictions_squeeze_func(X_test_adv)
            l1_dist_vec = model_eval_distance(predictions_orig, predictions_squeezed)
            l1_dist.append(l1_dist_vec)
        print("---Results are stored in ", csv_fpath, '\n')
        return l1_dist

    def calculate_l1_distance_jsma(sess, x, predictions, X_test, X_test_adv, csv_fpath):
        print("\n===Calculating L1 distance with feature squeezing...")
        nb_examples = len(X_test_adv)

        l1_dist = np.zeros((nb_examples, 2))

        # for i, k_width in enumerate(range(1,11)):
        for i, k_width in [(0, 3)]:
            X_test_adv_smoothed = median_filter_np(X_test_adv, k_width)
            X_test_smoothed = median_filter_np(X_test, k_width)

            l1_dist_vec = model_eval_distance_dual_input(sess, x, predictions, X_test, X_test_smoothed)
            l1_dist[:, 2 * i] = l1_dist_vec

            l1_dist_vec = model_eval_distance_dual_input(sess, x, predictions, X_test_adv, X_test_adv_smoothed)
            l1_dist[:, 2 * i + 1] = l1_dist_vec

        np.savetxt(csv_fpath, l1_dist, delimiter=',')
        print("---Results are stored in ", csv_fpath, '\n')
        return l1_dist

    def calculate_l1_distance_joint(sess, x, predictions, X_test, X_test_adv_fgsm, X_test_adv_jsma, csv_fpath):
        print("\n===Calculating max(L1) distance with feature squeezing...")
        nb_examples = max(len(X_test), len(X_test_adv_fgsm), len(X_test_adv_jsma))

        l1_dist = np.zeros((nb_examples, 3))
        median_filter_width = 3

        for i, X in enumerate([X_test, X_test_adv_fgsm, X_test_adv_jsma]):
            X_test1 = X
            X_test2 = median_filter_np(X_test1, median_filter_width)
            X_test3 = binary_filter_np(X_test1)

            l1_dist_vec = tf_model_eval_dist_tri_input(sess, x, predictions, X_test1, X_test2, X_test3, mode='max')
            l1_dist[:len(X), i] = l1_dist_vec

        np.savetxt(csv_fpath, l1_dist, delimiter=',')
        print("---Results are stored in ", csv_fpath, '\n')
        return l1_dist

    def detection(self, x):
        predictions = self.model(x)
        predictions_clip = self.model(torch.clamp(x, min=0.0, max=1.0))
