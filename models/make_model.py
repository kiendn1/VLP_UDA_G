import torch
import torch.nn as nn
from models.backbone import get_backbone
from models import cmkd
import logging
import torch.nn.functional as F
import copy
import numpy as np

_logger = logging.getLogger(__name__)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
    elif classname.find('BatchNorm') != -1:
        m.bias.requires_grad_(False)
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
       m.eval()
       
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

class TransferNet(nn.Module):
    def __init__(self, args, train=True):
        super(TransferNet, self).__init__()
        # define the network
        # get the feature extractor and the pretrained head
        self.args = args
        self.num_class = args.num_class
        self.base_network = get_backbone(args).cuda()
        self.teacher_model = copy.deepcopy(self.base_network)
        self.teacher_model.eval()

        # define the task head
        self.classifier_layer = nn.Sequential(
            nn.BatchNorm1d(self.base_network.output_num),
            nn.LayerNorm(self.base_network.output_num, eps=1e-6),
            nn.Linear(self.base_network.output_num, self.num_class,bias=False))
        self.classifier_layer.apply(weights_init_classifier)

        if train:
            # define the loss functions
            self.cmkd = cmkd.CMKD(args)
            self.clf_loss = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    def forward(self, args, source_img, gen_img, target_img, source_label, gen_label, target_strong=None, label_set=None):
        self.base_network.apply(fix_bn)
        source_feat = self.base_network.forward_features(source_img)

        # calculate source classification loss Lclf
        source_logits = self.classifier_layer(source_feat)
        source_clf_loss = self.clf_loss(source_logits, source_label)
        
        gen_feat = self.base_network.forward_features(gen_img)

        # calculate source classification loss Lclf
        gen_logits = self.classifier_layer(gen_feat)
        gen_clf_loss = self.clf_loss(gen_logits, gen_label)
        
        clf_loss = source_clf_loss + gen_clf_loss

        if not self.args.baseline:
            source_logits_clip = self.base_network.forward_head(source_feat)
            target_feat = self.base_network.forward_features(target_img)

            # calculate calibrated probability alignment loss Lcpa
            target_clip_logits = self.base_network.forward_head(target_feat)
            target_logits = self.classifier_layer(target_feat)

            # calculate calibrated gini impurity loss Lcgi
            transfer_loss, target_pred_mix = self.cmkd(target_logits, target_clip_logits, source_logits_clip, source_label,label_set)

        else:
            transfer_loss = torch.tensor(0).to(source_label.device)

        if self.args.fixmatch and target_strong is not None:
            target_pred = F.softmax(target_logits, dim=1)
            if label_set is not None:
                compl_label_set = list(set(torch.range(0, 64).tolist()) - set(label_set))
                compl_label_set = [int(item) for item in compl_label_set]
                target_pred[:, compl_label_set] = 0.0
            max_prob, pred_u = torch.max(target_pred, dim=-1)
            target_strong_feature = self.base_network.forward_features(target_strong)
            target_strong = self.classifier_layer(target_strong_feature)
            fixmatch_loss = self.args.fixmatch_factor * (F.cross_entropy(target_strong, pred_u.detach(), reduction='none') *
                                                         max_prob.ge(self.args.fixmatch_threshold).float().detach()).mean()

            target_pred_clip = F.softmax(target_clip_logits,dim=-1)
            if label_set is not None:
                target_pred_clip[:, compl_label_set] = 0.0

            max_prob, pred_u = torch.max(target_pred_clip, dim=-1)
            target_strong = self.base_network.forward_head(target_strong_feature)
            fixmatch_loss += self.args.fixmatch_factor * (
                        F.cross_entropy(target_strong, pred_u.detach(), reduction='none') *
                        max_prob.ge(self.args.fixmatch_threshold).float().detach()).mean()
            transfer_loss += fixmatch_loss

        if self.args.pda:
            clf_loss = 0.5 * clf_loss
            transfer_loss = 0.1 * transfer_loss
            
        if args.cutmix:
            r = np.random.rand(1)
            if args.beta > 0 and r < args.cutmix_prob:
                # generate mixed sample
                lam = np.random.beta(args.beta, args.beta)
                rand_index = torch.randperm(target_img.size()[0])[:16].cuda()
                target_a = gen_label
                target_b = target_pred_mix[rand_index]
                mix_img = gen_img.clone().detach().cuda()
                bbx1, bby1, bbx2, bby2 = rand_bbox(mix_img.size(), lam)
                mix_img[:, :, bbx1:bbx2, bby1:bby2] = target_img[rand_index, :, bbx1:bbx2, bby1:bby2]
                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (gen_img.size()[-1] * gen_img.size()[-2]))
                # compute output
                mix_output = self.base_network.forward_features(mix_img)
                mix_logits = self.classifier_layer(mix_output)
                mix_clf_loss = self.clf_loss(mix_logits, target_a) * lam + self.clf_loss(mix_logits, target_b) * (1. - lam)
                
                clf_loss = clf_loss + mix_clf_loss
        return clf_loss, transfer_loss

    def get_parameters(self, initial_lr=1.0):
        params=[
            {'params': self.base_network.model.visual.parameters(), 'lr': initial_lr},
            {'params': self.classifier_layer.parameters(), 'lr': self.args.multiple_lr_classifier * initial_lr}
]
        return params

    def predict(self, x):
        features = self.base_network.forward_features(x)
        logit = self.classifier_layer(features)
        return logit

    def clip_predict(self, x):
        logit = self.base_network(x)
        return logit