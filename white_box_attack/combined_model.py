from torch import nn
import copy
import torch
# method from paper <Detecting Adversarial Examples through Image Transformation>
# this model is only to generate adversarial examples
class CombinedModel(nn.Module):
    def  __init__(self, img_classifier, detector_network):
        super(CombinedModel, self).__init__()
        self.img_classifier = img_classifier # pretrained_model  如果攻击算法
        self.detector = detector_network  # pretrained_model
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        logits_img_class = self.img_classifier(x)  # output num class
        logits_det = self.detector(x) # 输出N,2 ，如果是正常样本则index=1大,否则index=0大
        soft_det = self.softmax(logits_det) # 每个logit的值都是0 -- 1之间
        Z_D = soft_det[:, 0]  # N index=0是表示被识别为噪音的概率，如果>0.5则表示是噪音
        Z_G = logits_img_class
        Z_D = Z_D * 2 * torch.max(Z_G,dim=1,keepdim=False)[0] # shape = (N,),如果
        Z_D = Z_D.unsqueeze(1) # shape = (N,1)
        Z_G = torch.cat([Z_G, Z_D],dim=1)  # N, class + 1
        return Z_G


class CombinedModelForNeuralFingerprint(nn.Module):
    def  __init__(self, img_classifier, detector_network, attacker):
        super(CombinedModelForNeuralFingerprint, self).__init__()
        self.img_classifier = img_classifier # pretrained_model  如果攻击算法
        self.detector = detector_network  # pretrained_model
        self.attacker = attacker

    def forward(self, x, img_gt, target):
        _, _, _, loss_fingerprint_dy = self.detector.get_all_loss(self.detector.model, x, img_gt, epoch=3)
        self.attacker.generate(x, img_gt, target, loss_fingerprint_dy)