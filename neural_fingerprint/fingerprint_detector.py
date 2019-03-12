from torch import nn
import numpy as np
import pickle
import torch.nn.functional as F
import torch
import os
from collections import defaultdict

from neural_fingerprint.fingerprint import Fingerprints,Example,Stats
from config import IMAGE_SIZE, IN_CHANNELS
import copy
from torch import optim

class NeuralFingerprintDetector(object):

    def __init__(self, dataset, model, num_dx, num_class, eps, out_fp_dxdy_dir):
        self.dataset = dataset
        self.model = model
        self.num_class = num_class
        self.dx_path = out_fp_dxdy_dir + '/fp_inputs_dx@{}.pkl'.format(dataset)
        self.dy_path = out_fp_dxdy_dir + '/fp_outputs_dy@{}.pkl'.format(dataset)
        self.num_dx = num_dx
        if os.path.exists(self.dx_path) and os.path.exists(self.dy_path):
            print("loading dx and dy from {} and {}".format(self.dx_path, self.dy_path))
            with open(self.dx_path, "rb") as file_obj:
                self.fp_dx = pickle.load(file_obj)
            with open(self.dy_path, "rb") as file_obj:
                self.fp_target = pickle.load(file_obj)
        else:
            self.fp_dx = [(np.random.rand(1,IN_CHANNELS[dataset],IMAGE_SIZE[dataset][0],IMAGE_SIZE[dataset][1])-0.5)*2*eps for i in range(num_dx)]
            self.fp_target = 0.254*np.ones((num_class, num_dx, num_class))
            for j in range(num_dx):
                for i in range(num_class):
                    self.fp_target[i, j, i] = - 0.7
            self.fp_target = 1.5 * self.fp_target
            self.dump_dx_dy()
        self.fp_target = torch.from_numpy(self.fp_target).cuda().float()

        self.fp = Fingerprints()
        self.fp.dx = self.fp_dx
        self.fp.dy = self.fp_target

        self.loss_func = nn.CrossEntropyLoss()
        self.loss_n = nn.MSELoss()
        self.verbose = True

    def dump_dx_dy(self):
        with open(self.dx_path, 'wb') as file_obj:
            pickle.dump(self.fp_dx, file_obj)
        with open(self.dy_path, "wb") as file_obj:
            pickle.dump(self.fp_target, file_obj)
        print("dump to {} and {} over".format(self.dx_path, self.dy_path))

    def train_one_image(self, model, x, y, optimizer, epoch=1):
        x, y = x.cuda(), y.cuda()

        real_bs = y.size(0)
        fp_target_var = torch.index_select(self.fp_target, 0, y)
        x_net = x
        for i in range(self.num_dx):
            dx = self.fp_dx[i]
            dx = torch.from_numpy(dx).float().cuda()
            x_net = torch.cat((x_net, x + dx))
        logits_net = model(x_net)
        output_net = F.log_softmax(logits_net)
        yhat = output_net[:real_bs]
        logits = logits_net[:real_bs]
        # 除以模长，归一化
        logits_norm = logits * torch.norm(logits, 2, 1, keepdim=True).reciprocal().expand(real_bs, self.num_class)
        loss_fingerprint_y = 0
        loss_fingerprint_dy = 0
        loss_vanilla = self.loss_func(yhat, y)
        for i in range(self.num_dx):
            fp_target_var_i = fp_target_var[:, i, :]
            logits_p = logits_net[(i + 1) * real_bs: (i + 2) * real_bs]
            logits_p_norm = logits_p * torch.norm(logits_p, 2, 1, keepdim=True).reciprocal().expand(real_bs,
                                                                                                    self.num_class)
            diff_logits_p = logits_p_norm - logits_norm + 0.00001
            loss_fingerprint_y += self.loss_n(logits_p_norm, fp_target_var_i)
            loss_fingerprint_dy += self.loss_n(diff_logits_p, fp_target_var_i)
        if self.dataset == "MNIST" or self.dataset == "F-MNIST":
            if epoch >= 1:
                loss = loss_vanilla + 1.0 * loss_fingerprint_dy
            else:
                loss = loss_vanilla
        else:
            if epoch >= 1:
                loss = loss_vanilla + (1.0 + 50.0 / self.num_dx) * loss_fingerprint_dy
            else:
                loss = loss_vanilla
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss, loss_vanilla, loss_fingerprint_y, loss_fingerprint_dy


    def train(self, epoch, optimizer, data_loader):
        self.model.train()
        for batch_idx, (x, y) in enumerate(data_loader):
            x,y = x.cuda(), y.cuda()
            loss, loss_vanilla, loss_fingerprint_y, loss_fingerprint_dy = self.train_one_image(self.model, x, y, optimizer, epoch)
            if batch_idx % 1000 == 0:
                print(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss vanilla: {:.3f} fp-y: {:.3f} fp-dy: {:.3f} Total Loss: {:.3f}'.format(
                        epoch, batch_idx * len(x), len(data_loader.dataset),
                               100. * batch_idx / len(data_loader),
                        loss_vanilla.item(),
                        loss_fingerprint_y.item(),
                        loss_fingerprint_dy.item(),
                        loss.item()))

    def test(self, epoch, data_loader, test_length=None):
        self.model.eval()
        test_loss = 0
        correct = 0
        loss_y = 0
        loss_dy = 0
        num_same_argmax = 0
        with torch.no_grad():
            for e,(data, target) in enumerate(data_loader):
                data, target = data.cuda(), target.cuda()
                data.requires_grad = False
                data_np = data.detach().cpu().numpy()
                real_bs = data_np.shape[0]
                logits = self.model(data)
                output = F.log_softmax(logits, dim=1)
                logits_norm = logits * torch.norm(logits, 2, 1, keepdim=True).reciprocal().expand(real_bs, self.num_class)
                fp_target_var = torch.index_select(self.fp_target, 0, target)
                for i in range(self.num_dx):
                    dx = self.fp_dx[i]
                    fp_target_var_i = fp_target_var[:, i, :]
                    logits_p = self.model(data + torch.from_numpy(dx).float().cuda())
                    logits_p_norm = logits_p * torch.norm(logits_p, 2, 1, keepdim=True).reciprocal().expand(real_bs,
                                                                                                            self.num_class)
                    diff = logits_p_norm - logits_norm
                    diff_class = diff.max(1, keepdim=True)[1]
                    fp_target_class = fp_target_var_i.max(1, keepdim=True)[1]
                    loss_y += self.loss_n(logits_p_norm, fp_target_var_i)
                    loss_dy += 10.0 * self.loss_n(diff, fp_target_var_i)
                    num_same_argmax += torch.sum(diff_class == fp_target_class)
                test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).detach().cpu().sum()
        if test_length is None:
            test_length = len(data_loader.dataset)
        test_loss /= test_length
        loss_y /= test_length
        loss_dy /= test_length
        argmax_acc = num_same_argmax.item() * 1.0 / (test_length * self.num_dx)
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, test_length,
            100. * correct / test_length))
        print('Fingerprints (on test): L(fp, y) loss: {:.4f}, L(fp, dy) loss: {:.4f}, argmax y = argmax f(x+dx) Accuracy: {}/{} ({:.0f}%)'.format(
                loss_y.item(), loss_dy.item(), num_same_argmax,
                len(data_loader.dataset) * self.num_dx,
                100. * argmax_acc))
        result = {"epoch": epoch,
                  "test-loss": test_loss,
                  "test-correct": correct,
                  "test-N": test_length,
                  "test-acc": correct / test_length,
                  "fingerprint-loss (y)": loss_y.item(),
                  "fingerprint-loss (dy)": loss_dy.item(),
                  "fingerprint-loss (argmax)": argmax_acc,
                  }
        return result

    def get_pr_wrapper(self, results, pos_names, neg_names, reject_thresholds, args):

        for e, _type in enumerate(["raw", "cond_correct"]):

            pr_results = {}

            for tau in reject_thresholds:

                tmp_results = {k: v[tau][e] for k, v in results.items()}

            pr_auc = get_pr_auc(pr_results, args, plot=True,
                                plot_name="{}-{}-{}".format(_type, "-".join(pos_names), "-".join(neg_names)))

            print(pos_names, neg_names, _type, "{}: AUC ROC {} PR {}".format(_type, roc_auc, pr_auc))

            pr_results["pr_auc"] = pr_auc

            # print("count stats")
            # for k,v in pr_results.items():
            #     print(k,v)

            path = os.path.join(args.log_dir,
                                "rates-roc-pr-auc_{}_{}_{}_tau_{:.4f}.pkl".format(_type, "-".join(pos_names),
                                                                                  "-".join(neg_names), tau))
            print("Saving pr result in", path)
            pickle.dump(pr_results, open(path, "wb"))


    def model_with_fingerprint(self, model, x, fp):
        # x : B x C x W x H with B = 1
        # Check y' = f(x+dx) for all dx

        assert x.size()[0] == 1  # batch

        # Get perturbations for predicted class

        logits = model(x)

        yhat = F.softmax(logits, dim=1)
        y_class = yhat.data.max(1, keepdim=True)[1]
        y_class = y_class.detach().cpu().numpy()[0, 0]

        # fixed_dxs : num_perturb x C x W x H
        fixed_dxs = torch.from_numpy(np.concatenate(fp.dx, axis=0)).float().cuda()

        # cmopute x + dx : broadcast! num_perturb x C x W x H
        xp = x + fixed_dxs

        # if args.debug: print("xp", xp.size(), "x", x.size(), "fixed_dxs", fixed_dxs.size())

        logits_p = self.model(xp)
        yhat_p = F.softmax(logits_p, dim=1)

        # compute f(x + dx) : num_perturb x num_class

        # print("get fixed_dys : num_target_class x num_perturb x num_class: for each target class, a set of perturbations and desired outputs (num_class).")
        fixed_dys = fp.dy
        logits_norm = logits * torch.norm(logits, 2, 1, keepdim=True).reciprocal().expand(1, self.num_class)
        logits_p_norm = logits_p * torch.norm(logits_p, 2, 1, keepdim=True).reciprocal().expand(self.num_dx,
                                                                                                self.num_class)
        diff_logits_p = logits_p_norm - logits_norm
        # diff_logits_p = diff_logits_p * torch.norm(diff_logits_p, 2, 1, keepdim=True).reciprocal().expand(args.num_dx, args.num_class)

        diff = fixed_dys - diff_logits_p

        diff_norm = torch.norm(diff, 2, dim=2)

        diff_norm = torch.mean(diff_norm, dim=1)

        y_class_with_fp = diff_norm.min(0, keepdim=True)[1]
        y_class_with_fp = y_class_with_fp.detach().cpu().numpy()[0]

        ex = Example(x, yhat, y_class)
        ex.dxs = fixed_dxs
        ex.yhat_p = yhat_p
        ex.diff = diff
        ex.diff_norm = diff_norm
        ex.y_class_with_fp = y_class_with_fp

        return ex

    def detect_with_fingerprints(self, ex, stats_per_tau):
        diff_norm = ex.diff_norm
        y_class_with_fp = ex.y_class_with_fp

        for reject_threshold, stats in stats_per_tau.items():

            stats.ids.add(ex.id)
            # Check legal: ? D({f(x+dx)}, {y^k}) < tau for all classes k.
            below_threshold = diff_norm < reject_threshold
            below_threshold_t = below_threshold[y_class_with_fp]  # 寻找diff向量中小于阈值的卡住，最小的元素对应的值是多少
            below_threshold_t = below_threshold_t.detach().cpu()
            is_legal = below_threshold_t.item() > 0
            ex.is_legal = is_legal

            if ex.is_legal:
                stats.ids_legal.add(ex.id)

            if ex.y == ex.y_class:  # ex.y is ground truth, 1 real image, 0 adv image
                stats.ids_correct.add(ex.id)


            if ex.y == ex.y_class_with_fp: # ex.y is ground truth
                stats.ids_correct_fp.add(ex.id)

            if ex.y_class == ex.y_class_with_fp:
                stats.ids_agree.add(ex.id)
                if ex.y == 1 and ex.is_legal:
                    stats.TP += 1
                elif ex.y == 0 and ex.is_legal:
                    stats.FP += 1
            else:  # predict as adv = 0
                if ex.y == 1 and ex.is_legal:
                    stats.FN += 1
                elif ex.y == 0 and ex.is_legal:
                    stats.TN += 1





        return ex, stats_per_tau


    def eval_with_fingerprints(self, data_loader, ds_name, reject_thresholds, test_results_by_tau, name):
        self.model.eval()
        stats_per_tau = {i: Stats(tau=i, name=name, ds_name=ds_name) for i in reject_thresholds}
        i = 0
        for e, (x, y) in enumerate(data_loader):
            data_np = x.detach().cpu().numpy()
            real_bs = data_np.shape[0]
            for b in range(real_bs):
                ex = self.model_with_fingerprint(self.model, x[b:b + 1], self.fp)
                # Careful! Needs Dataloader with shuffle=False
                ex.id = i
                ex.y = y[b]
                ex, stats_per_tau = self.detect_with_fingerprints(ex, stats_per_tau)
                i += 1
                if self.verbose:
                    print("\nx", x[b:b + 1].size(), "y", y[b:b + 1], y[b:b + 1].size())
                    print("Fingerprinting image (hash:", hash(x[b:b + 1]), ") class", y[b])
                    print("Model    class prediction: [", ex.y_class, "] from logits:", ex.yhat)
                    print("Model+fp class prediction: [{}] from diff_norm: {}".format(ex.y_class_with_fp,
                                                                                      ex.diff_norm.detach().cpu().numpy()))
            if e % 10 == 0:
                print("Ex: {} batch {} of size {}".format(i, e, real_bs))
            if e >= 0:
                break
        results = defaultdict(lambda: None)
        stats_results = defaultdict(lambda: None)

        for tau, stats in stats_per_tau.items():
            stats.counts = stats.compute_counts()
            # use test x for which f(x) was correct. If the current dataset is not test, we need an external set of ids.
            if test_results_by_tau:
                ids_correct = test_results_by_tau[tau].ids_correct
            else:
                continue
            stats.counts_correct = stats.compute_counts(ids_correct=ids_correct)
            if self.verbose:
                print("Stats raw (tau {})".format(tau))
                stats.show(stats.counts)
                print("Stats cond_correct (tau {})".format(tau))
                stats.show(stats.counts_correct)
            results[tau] = [stats.counts, stats.counts_correct]
            stats_results[tau] = stats
        return results, stats_results

    def eval_with_fingerprints_finetune(self, val_loader, ds_name, reject_thresholds, test_results_by_tau, num_updates, lr):
        test_net = copy.deepcopy(self.model)
        stats_per_tau = {i: Stats(tau=i, name=ds_name, ds_name=ds_name) for i in reject_thresholds}
        i = 0
        all_F1_scores = []
        all_accuracys = []
        all_tau = []
        for support_images, support_labels, query_images, query_labels, positive_labels in val_loader:

            positive_labels = positive_labels.detach().cpu().numpy()
            support_labels = support_labels.detach().cpu().numpy()
            query_labels = query_labels.detach().cpu().numpy()
            support_labels = (support_labels == positive_labels).astype(np.int64)
            query_labels = (query_labels == positive_labels).astype(np.int64)
            support_images = support_images.cuda()
            query_images = query_images.cuda()
            support_labels = torch.from_numpy(support_labels).cuda()
            query_labels = torch.from_numpy(query_labels).cuda()

            for task_idx in range(support_images.size(0)):
                test_net.copy_weights(self.model)
                optimizer = optim.Adam(test_net.parameters(), lr=lr)
                x = support_images[task_idx]  # FIXME finetune应该只用干净的图片来fine-tune
                y = support_labels[task_idx]
                batch_size = x.size(0)
                x = x.view(batch_size, IN_CHANNELS[ds_name], IMAGE_SIZE[ds_name][0], IMAGE_SIZE[ds_name][1])
                for _ in range(num_updates):  # 先fine_tune
                    loss, _, _, _ = self.train_one_image(test_net, x,y, optimizer, 1)
                test_net.eval()
                x, y = query_images[task_idx], query_labels[task_idx]
                batch_size = x.size(0)
                x = x.view(batch_size, IN_CHANNELS[ds_name], IMAGE_SIZE[ds_name][0], IMAGE_SIZE[ds_name][1])

                data_np = x.detach().cpu().numpy()
                real_bs = data_np.shape[0]
                predict_label_list = []
                gt_label_list = []

                for b in range(real_bs):
                    ex = self.model_with_fingerprint(test_net, x[b:b + 1], self.fp)
                    # Careful! Needs Dataloader with shuffle=False
                    ex.id = i
                    ex.y = y[b].item() # ground truth of real image : 1 , adversarial image 0
                    gt_label_list.append(ex.y)

                    ex, stats_per_tau = self.detect_with_fingerprints(ex, stats_per_tau)
                    i += 1

                # results = defaultdict(lambda: None)
                # stats_results = defaultdict(lambda: None)

                result = {}
                for tau, stats in stats_per_tau.items():
                    precision_tau = 0
                    recall_tau = 0
                    accuracy_tau = 0
                    F1_score_tau = 0
                    if stats.TP + stats.FP != 0:
                        precision_tau = stats.TP / float(stats.TP + stats.FP)
                    if stats.TP + stats.FN != 0:
                        recall_tau = stats.TP / float(stats.TP + stats.FN)
                    if stats.TP + stats.TN + stats.FP + stats.FN != 0:
                        accuracy_tau = (stats.TP + stats.TN) /float(stats.TP + stats.TN + stats.FP + stats.FN)
                    if precision_tau + recall_tau != 0:
                        F1_score_tau = 2 * precision_tau * recall_tau / (precision_tau + recall_tau)
                    result[tau] = (accuracy_tau, F1_score_tau)

                best_F1 = max(list(result.values()), key=lambda e:e[1])[1]
                best_acc = max(list(result.values()), key=lambda e:e[0])[0]
                best_tau = max(list(result.items()), key=lambda e:e[1][0])[0]
                all_accuracys.append(best_acc)
                all_F1_scores.append(best_F1)
                all_tau.append(best_tau)
                stats_per_tau.clear()
                for i in reject_thresholds:
                    stats_per_tau[i] = Stats(tau=i, name=ds_name, ds_name=ds_name)
            print("process 100 task done, current last task's acc:{} F1:{}".format(best_acc, best_F1))
        accuracy = np.mean(all_accuracys)
        F1 = np.mean(all_F1_scores)
        tau = np.mean(best_tau)
        print("final accuracy: {},  F1: {}  tau:{}".format(accuracy, F1, tau))
        return accuracy, F1
