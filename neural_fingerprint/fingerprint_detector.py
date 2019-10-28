import time

from torch import nn
import numpy as np
import pickle
import torch.nn.functional as F
import torch
import os
from collections import defaultdict

from neural_fingerprint.fingerprint import Fingerprints,Example,Stats
from config import IMAGE_SIZE, IN_CHANNELS, META_ATTACKER_INDEX
import copy
from torch import optim
class NeuralFingerprintDetector(object):

    def __init__(self, dataset, model, num_dx, num_class, eps, out_fp_dxdy_dir):
        self.dataset = dataset
        self.model = model
        self.num_class = num_class
        self.dx_path = out_fp_dxdy_dir + '/fp_inputs_dx@num_dx_{}_eps_{}@{}.pkl'.format(num_dx, eps, dataset)
        self.dy_path = out_fp_dxdy_dir + '/fp_outputs_dy@num_dx_{}_num_class_{}@{}.pkl'.format(num_dx, num_class, dataset)
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


    def get_all_loss(self, model, x,y, epoch):
        x, y = x.cuda(), y.cuda()

        real_bs = y.size(0)
        fp_target_var = torch.index_select(self.fp_target, 0, y)
        x_net = x
        for i in range(self.num_dx):
            dx = self.fp_dx[i]
            dx = torch.from_numpy(dx).float().cuda()
            x_net = torch.cat((x_net, x + dx))
        logits_net = model(x_net)
        output_net = F.log_softmax(logits_net, dim=1)
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
        return loss, loss_vanilla, loss_fingerprint_y, loss_fingerprint_dy

    def train_one_image(self, model, x, y, optimizer, epoch=1):
        loss, loss_vanilla, loss_fingerprint_y, loss_fingerprint_dy = self.get_all_loss(model,x,y, epoch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss, loss_vanilla, loss_fingerprint_y, loss_fingerprint_dy


    def train(self, epoch, optimizer, data_loader):
        self.model.train()
        for batch_idx, (x, y) in enumerate(data_loader):
            x,y = x.cuda(), y.cuda()
            loss, loss_vanilla, loss_fingerprint_y, loss_fingerprint_dy = self.train_one_image(self.model, x, y, optimizer, epoch)
            if batch_idx % 100 == 0:
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
        correct_fp = 0
        fingerprint_accuracy = []
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
                pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).detach().cpu().sum()
        if test_length is None:
            test_length = len(data_loader.dataset)
        test_loss /= test_length
        loss_y /= test_length
        loss_dy /= test_length
        argmax_acc = num_same_argmax * 1.0 / (test_length * self.num_dx)
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

        diff = fixed_dys - diff_logits_p # 论文公式9

        diff_norm = torch.norm(diff, 2, dim=2)

        diff_norm = torch.mean(diff_norm, dim=1)  #论文公式9求均值

        y_class_with_fp = diff_norm.min(0, keepdim=True)[1] #差距最小的那个logit的位置
        y_class_with_fp = y_class_with_fp.detach().cpu().numpy()[0]

        ex = Example(x, yhat, y_class)
        ex.dxs = fixed_dxs
        ex.yhat_p = yhat_p
        ex.diff = diff
        ex.diff_norm = diff_norm.detach().cpu().numpy()
        ex.y_class_with_fp = y_class_with_fp

        return ex

    def detect_with_fingerprints(self, ex, stats_per_tau):
        diff_norm = ex.diff_norm
        y_class_with_fp = ex.y_class_with_fp

        for reject_threshold, stats in stats_per_tau.items():

            stats.ids.add(ex.id)
            # Check legal: ? D({f(x+dx)}, {y^k}) < tau for all classes k.
            below_threshold = diff_norm < reject_threshold
            below_threshold = below_threshold.astype(np.int32)
            below_threshold_t = below_threshold[y_class_with_fp]

            is_legal = below_threshold_t > 0
            ex.is_legal = is_legal

            if ex.y == ex.y_class:  # y是gt label  y_class模型f(x)输出
                stats.ids_correct.add(ex.id)

            if ex.y == ex.y_class_with_fp:
                stats.ids_correct_fp.add(ex.id)

            if ex.y_class == ex.y_class_with_fp:
                stats.ids_agree.add(ex.id)

            if ex.binary_y == 1:
                stats.P.add(ex.id)
                stats.ground_truth_list.append(1)
            else:
                stats.N.add(ex.id)
                stats.ground_truth_list.append(0)
            if is_legal:
                stats.predition_list.append(1)
            else:
                stats.predition_list.append(0)
            if is_legal and ex.binary_y == 1:
                stats.TP.add(ex.id)
            if is_legal and ex.binary_y == 0:
                stats.FP.add(ex.id)
            if ex.is_legal:
                stats.ids_legal.add(ex.id)

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

        results = defaultdict(lambda: None)
        stats_results = defaultdict(lambda: None)

        for tau, stats in stats_per_tau.items():
            stats.counts = stats.compute_counts()  # 这个字典里的num_correct_fp就是预测正确的个数
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



    def eval_with_fingerprints_finetune(self, val_loader, ds_name, reject_thresholds, num_updates, lr):
        test_net = copy.deepcopy(self.model)
        stats_per_tau = {thresh: Stats(tau=thresh, name=ds_name, ds_name=ds_name) for thresh in reject_thresholds}
        i = 0
        all_F1_scores = []
        all_tau = []
        # 注意这个val_loader要特别定制化
        each_attack_stats = val_loader.dataset.fetch_attack_name
        attacker_stats = defaultdict(list)
        for pack in val_loader:
            if each_attack_stats:
                support_images, support_gt_labels, support_binary_labels, query_images, query_gt_labels, query_binary_labels, adversary_indexes, _ = pack
            else:
                support_images, support_gt_labels, support_binary_labels, query_images, query_gt_labels, query_binary_labels, _ = pack
            support_binary_labels = support_binary_labels.detach().cpu().numpy()
            support_gt_labels = support_gt_labels.cuda()
            support_images = support_images.cuda()
            query_images = query_images.cuda()
            for task_idx in range(support_images.size(0)):

                clean_support_index = np.where(support_binary_labels[task_idx] == 1)[0]
                clean_imgs = support_images[task_idx][clean_support_index]
                clean_labels = support_gt_labels[task_idx][clean_support_index]  # support label 需要传入干净图 的真正label，而不是0/1
                test_net.copy_weights(self.model)

                optimizer = optim.SGD(test_net.parameters(), lr=lr)
                batch_size = clean_imgs.size(0)
                clean_imgs = clean_imgs.view(batch_size, IN_CHANNELS[ds_name], IMAGE_SIZE[ds_name][0], IMAGE_SIZE[ds_name][1])
                test_net.train()
                for m in test_net.modules():
                    if isinstance(m, torch.nn.BatchNorm2d):
                        m.eval()  # BN层会出问题，不要训练
                for _ in range(num_updates):  # 先fine_tune
                    self.train_one_image(test_net, clean_imgs, clean_labels, optimizer, 1)

                test_net.eval()
                x, y = query_images[task_idx], query_gt_labels[task_idx]  # 注意这个分类信息是img gt label
                binary_y = query_binary_labels[task_idx]
                y = y.detach().cpu().numpy()
                binary_y = binary_y.detach().cpu().numpy()
                batch_size = x.size(0)
                x = x.view(batch_size, IN_CHANNELS[ds_name], IMAGE_SIZE[ds_name][0], IMAGE_SIZE[ds_name][1])

                data_np = x.detach().cpu().numpy()
                real_bs = data_np.shape[0]

                for b in range(real_bs):
                    ex = self.model_with_fingerprint(test_net, x[b:b + 1], self.fp)
                    # Careful! Needs Dataloader with shuffle=False
                    ex.id = i
                    ex.y = y[b] # ground truth of real image : 1 , adversarial image 0
                    ex.binary_y = binary_y[b]
                    ex, stats_per_tau = self.detect_with_fingerprints(ex, stats_per_tau)
                    i += 1

                # results = defaultdict(lambda: None)
                # stats_results = defaultdict(lambda: None)

                result = {}
                for tau, stats in stats_per_tau.items():
                    stats.counts = stats.compute_counts()
                    ids_correct = stats.ids_correct
                    F1 = stats.compute_counts(ids_correct=ids_correct)["F1"]
                    result[tau] = F1
                # best_F1 = result[min(result.keys(), key=lambda k: abs(k-threshold))]
                best_F1 = max(list(result.values()))
                # best_tau = threshold
                best_tau = max(list(result.items()), key=lambda e:e[1])[0]
                if each_attack_stats:
                    adversary = META_ATTACKER_INDEX[adversary_indexes[task_idx].item()]
                    attacker_stats[adversary].append(best_F1)

                all_F1_scores.append(best_F1)
                all_tau.append(best_tau)
                stats_per_tau.clear()
                for thresh in reject_thresholds:
                    stats_per_tau[thresh] = Stats(tau=thresh, name=ds_name, ds_name=ds_name)
                print("evaluate_accuracy task {}, F1:{}".format(task_idx, best_F1))
        F1 = np.mean(all_F1_scores)
        tau = np.mean(all_tau)
        for adversary, query_F1_score_list in attacker_stats.items():
            attacker_stats[adversary] = np.mean(query_F1_score_list)


        del test_net
        print("final   F1: {}  tau:{}".format( F1, tau))
        return F1, tau, attacker_stats



    def test_speed(self, val_loader, ds_name, reject_thresholds, num_updates, lr):
        test_net = copy.deepcopy(self.model)
        stats_per_tau = {thresh: Stats(tau=thresh, name=ds_name, ds_name=ds_name) for thresh in reject_thresholds}
        i = 0
        all_times = []
        # 注意这个val_loader要特别定制化
        for support_images,support_gt_labels, support_binary_labels, query_images, query_gt_labels, query_binary_labels,_ in val_loader:
            support_binary_labels = support_binary_labels.detach().cpu().numpy()
            support_gt_labels = support_gt_labels.cuda()
            support_images = support_images.cuda()
            query_images = query_images.cuda()
            for task_idx in range(support_images.size(0)):
                print("evaluate_accuracy task {}".format(task_idx))
                clean_support_index = np.where(support_binary_labels[task_idx] == 1)[0]
                clean_imgs = support_images[task_idx][clean_support_index]
                clean_labels = support_gt_labels[task_idx][clean_support_index]  # support label 需要传入干净图 的真正label，而不是0/1
                test_net.copy_weights(self.model)

                optimizer = optim.SGD(test_net.parameters(), lr=lr)
                batch_size = clean_imgs.size(0)
                clean_imgs = clean_imgs.view(batch_size, IN_CHANNELS[ds_name], IMAGE_SIZE[ds_name][0], IMAGE_SIZE[ds_name][1])
                test_net.train()
                before_time = time.time()
                for m in test_net.modules():
                    if isinstance(m, torch.nn.BatchNorm2d):
                        m.eval()  # BN层会出问题，不要训练
                for _ in range(num_updates):  # 先fine_tune
                    self.train_one_image(test_net, clean_imgs, clean_labels, optimizer, 1)

                test_net.eval()
                x, y = query_images[task_idx], query_gt_labels[task_idx]  # 注意这个分类信息是img gt label
                binary_y = query_binary_labels[task_idx]
                y = y.detach().cpu().numpy()
                binary_y = binary_y.detach().cpu().numpy()
                batch_size = x.size(0)
                x = x.view(batch_size, IN_CHANNELS[ds_name], IMAGE_SIZE[ds_name][0], IMAGE_SIZE[ds_name][1])

                data_np = x.detach().cpu().numpy()
                real_bs = data_np.shape[0]

                for b in range(real_bs):
                    ex = self.model_with_fingerprint(test_net, x[b:b + 1], self.fp)
                    # Careful! Needs Dataloader with shuffle=False
                    ex.id = i
                    ex.y = y[b] # ground truth of real image : 1 , adversarial image 0
                    ex.binary_y = binary_y[b]
                    ex, stats_per_tau = self.detect_with_fingerprints(ex, stats_per_tau)
                    i += 1

                # results = defaultdict(lambda: None)
                # stats_results = defaultdict(lambda: None)

                result = {}
                for tau, stats in stats_per_tau.items():
                    stats.counts = stats.compute_counts()
                    ids_correct = stats.ids_correct
                    F1 = stats.compute_counts(ids_correct=ids_correct)["F1"]
                    result[tau] = F1
                # best_F1 = result[min(result.keys(), key=lambda k: abs(k-threshold))]
                best_F1 = max(list(result.values()))
                # best_tau = threshold
                time_elapse = time.time() - before_time
                best_tau = max(list(result.items()), key=lambda e:e[1])[0]

                all_times.append(time_elapse)
                stats_per_tau.clear()
                for thresh in reject_thresholds:
                    stats_per_tau[thresh] = Stats(tau=thresh, name=ds_name, ds_name=ds_name)
        mean_time = np.mean(all_times)
        var_time = np.var(all_times)
        del test_net
        return mean_time, var_time