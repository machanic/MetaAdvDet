"""
Created on Fri Dec 16 01:24:11 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""

import torch
from PIL import Image
from torch import nn



class IterativeFastGradientSignUntargeted(object):
    """
        This class is similar to BasicIterativeMethodUntargeted
         but its goal is to create a adversarial example whose prediction is different from the original prediction label.
        It minimizes the initial class activation with iterative grad sign updates
    """
    def __init__(self, model, alpha=0.01, max_iters=10, lambd=0.5):
        self.model = model
        self.model.eval()
        # Movement multiplier per iteration
        self.alpha = alpha
        # Create the folder to export images if not exists
        self.max_iters = max_iters
        self.lambd = lambd

    def generate(self, x, orig_class, target, addition_loss=None):
        # Define loss functions
        ce_loss = nn.CrossEntropyLoss()
        # Process image
        processed_image = x.detach().cuda()
        # Start iteration
        for i in range(self.max_iters):
            # zero_gradients(x)
            # Zero out previous gradients
            # Can also use zero_gradients(x)
            last_proceed_img = processed_image
            processed_image = processed_image.detach()
            del last_proceed_img
            processed_image.requires_grad = True
            processed_image.grad = None
            self.model.zero_grad()
            # Forward pass
            out = self.model(processed_image)
            # Calculate CE loss
            pred_loss = ce_loss(out, orig_class)
            if addition_loss is not None:
                pred_loss += self.lambd * addition_loss
            # Do backward pass
            pred_loss.backward()
            # Create Noise
            # Here, processed_image.grad.data is also the same thing is the backward gradient from
            # the first layer, can use that with hooks as well
            adv_noise = self.alpha * torch.sign(processed_image.grad)
            # Add Noise to processed image
            processed_image = processed_image +  adv_noise

            # Confirming if the image is indeed adversarial with added noise
            # This is necessary (for some cases) because when we recreate image
            # the values become integers between 1 and 255 and sometimes the adversariality
            # is lost in the recreation process
            # if gen_correct:
            #     break
            del out
            del pred_loss
        return processed_image


class IterativeFastGradientSignTargetedFingerprint(object):
    """
        The Basic Iterative Method is the iterative version of FGM and FGSM. If target labels are not specified, the
        attack aims for the least likely class (the prediction with the lowest score) for each input.
        Paper link: https://arxiv.org/abs/1607.02533
        It maximizes the target class activation with iterative grad sign updates
    """
    def __init__(self, model, alpha, max_iters=10, neural_fp=None):
        self.model = model
        self.model.eval()
        self.max_iters = max_iters
        # Movement multiplier per iteration
        self.alpha = alpha
        self.lambd = 1.0
        self.neural_fp = neural_fp
        # Create the folder to export images if not exists

    def generate(self, x, orig_class, target):
        # I honestly dont know a better way to create a variable with specific value
        # Targeting the specific class
        # Define loss functions
        ce_loss = nn.CrossEntropyLoss()
        # Process image
        processed_image = x.detach().cuda()
        # Start iteration
        gen_correct = False
        for i in range(self.max_iters):
            # zero_gradients(x)
            # Zero out previous gradients
            # Can also use zero_gradients(x)
            last_proceed_img = processed_image
            processed_image = processed_image.detach()
            del last_proceed_img
            processed_image.requires_grad = True
            processed_image.grad = None
            self.model.zero_grad()
            # Forward pass
            out = self.model(processed_image)
            # Calculate CE loss
            pred_loss = ce_loss(out, target)
            _, _, _, loss_fingerprint_dy =self.neural_fp.get_all_loss(self.neural_fp.model, processed_image, orig_class, epoch=3)
            pred_loss += self.lambd * loss_fingerprint_dy
            # Do backward pass
            pred_loss.backward()
            # Create Noise
            # Here, processed_image.grad.data is also the same thing is the backward gradient from
            # the first layer, can use that with hooks as well
            adv_noise = self.alpha * torch.sign(processed_image.grad)
            # Add noise to processed image
            processed_image = processed_image - adv_noise

            # Confirming if the image is indeed adversarial with added noise
            # This is necessary (for some cases) because when we recreate image
            # the values become integers between 1 and 255 and sometimes the adversariality
            # is lost in the recreation process
            del out
            del pred_loss
        return processed_image
