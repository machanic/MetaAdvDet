import os
import pickle


def adaptive_fgsm(model, x, fingerprint_dir, eps, clip_min=None, clip_max=None,
                target=None, model_logits = None, alpha = None, dataset=None):
    """
        Computes symbolic TF tensor for the adversarial samples. This must
        be evaluated with a session.run call.
        :param x: the input placeholder
        :param eps: the epsilon (input variation parameter)
        :param clip_min: optional parameter that can be used to set a minimum
                        value for components of the example returned
        :param clip_max: optional parameter that can be used to set a maximum
                        value for components of the example returned
        :param y: the output placeholder. Use None (the default) to avoid the
                label leaking effect.
        :return: a tensor for the adversarial example
        """
    fixed_dxs = pickle.load(open(os.path.join(fingerprint_dir, "fp_inputs_dx.pkl"), "rb"))
    fixed_dys = pickle.load(open(os.path.join(fingerprint_dir, "fp_outputs.pkl"), "rb"))
    y = model(x)
    pred_class = y.max(1)[1]
    loss_fp = 0
    [a, b, c] = np.shape(fixed_dys)
    num_dx = b
