import pickle

import matplotlib.pyplot as plt
import numpy as np
import json


def read_json_data(json_path):
    x = []
    y = []
    with open(json_path, "r") as file_obj:
        data = json.load(file_obj)["CIFAR-10"]
        for key, val_json in sorted(data.items(), key=lambda e:int(e[0])):
            x.append(int(key))
            y.append(val_json["query_F1"])
    return np.array(x), np.array(y) * 100


def read_deep_learning_shots_data(json_path):
    x_no_balance = []
    y_no_balance = []
    x_balance = []
    y_balance = []
    with open(json_path, "r") as file_obj:
        data = json.load(file_obj)["CIFAR-10_no_balance"]
        for key, val_json in sorted(data.items(), key=lambda e:int(e[0])):
            x_no_balance.append(int(key))
            y_no_balance.append(val_json["query_F1"])

    with open(json_path, "r") as file_obj:
        data = json.load(file_obj)["CIFAR-10_balance"]
        for key, val_json in sorted(data.items(), key=lambda e:int(e[0])):
            x_balance.append(int(key))
            y_balance.append(val_json["query_F1"])

    return np.array(x_no_balance), np.array(y_no_balance) * 100, np.array(x_balance), np.array(y_balance) * 100


def read_deep_learning_finetune_data(json_path):
    x_balance = []
    y_balance = []
    x_no_balance = []
    y_no_balance = []
    with open(json_path, "r") as file_obj:
        data = json.load(file_obj)["CIFAR-10_no_balance"]
        for key, val_json in sorted(data.items(), key=lambda e:int(e[0])):
            x_no_balance.append(int(key))
            y_no_balance.append(val_json["query_F1"])
    with open(json_path, "r") as file_obj:
        data = json.load(file_obj)["CIFAR-10_balance"]
        for key, val_json in sorted(data.items(), key=lambda e:int(e[0])):
            x_balance.append(int(key))
            y_balance.append(val_json["query_F1"])

    return np.array(x_no_balance), np.array(y_no_balance) * 100, np.array(x_balance), np.array(y_balance) * 100


def draw_shots_line_curve(x_list, y_list, save_fig_name):
    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=(8, 6))
    x_meta=  x_list[0]
    x_deep_nobalance = x_list[1]
    x_deep_balance = x_list[2]
    y_meta = y_list[0]
    y_deep_nobalance = y_list[1]
    y_deep_balance = y_list[2]
    line, = plt.plot(x_meta, y_meta, marker='.', color="r", label="AdvMetaDet (ours)")
    line, = plt.plot(x_deep_nobalance, y_deep_nobalance, marker='.', color="b", label="DNN")
    line, = plt.plot(x_deep_balance, y_deep_balance,marker='.',color='y', label="DNN(balanced)")
    plt.xlim(0, 16)
    plt.ylim(50, 90)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], fontsize=15)
    plt.yticks([50,55,60,65,70,75,80,85,90], fontsize=15)
    plt.xlabel("shots", fontsize=18)
    plt.ylabel("F1 score of query set(%)", fontsize=18)
    plt.legend(loc='upper left', prop={'size': 17})
    plt.savefig(save_fig_name, dpi=200)


def draw_tasks_line_curve(x, y, save_fig_name):
    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=(8, 6))
    line, = plt.plot(x, y, marker='.', color="r")
    plt.xlim(4, x[-1] + 1)
    plt.ylim(55, 65)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.xticks(x, fontsize=15)
    plt.yticks([55, 58, 60,62, 65], fontsize=15)
    plt.xlabel("task number", fontsize=18)
    plt.ylabel("F1 score of query set(%)", fontsize=18)
    plt.legend(loc='upper right', prop={'size': 17})
    plt.savefig(save_fig_name, dpi=200)

def draw_inner_update_line_curve(x_meta, y_meta, save_fig_name):
    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=(8, 6))
    line, = plt.plot(x_meta, y_meta, marker='.', color="r", )
    plt.xlim(4, x_meta[-1] + 1)
    plt.ylim(60,64)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.xticks(x_meta, fontsize=15)
    plt.yticks([60,61,62,63,64,65], fontsize=15)
    plt.xlabel("inner update times", fontsize=18)
    plt.ylabel("F1 score of query set(%)", fontsize=18)
    plt.legend(loc='upper right', prop={'size': 17})
    plt.savefig(save_fig_name, dpi=200)

def draw_finetune_line_curve(x_list, y_list, save_fig_name):
    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=(8, 6))
    x_meta = x_list[0]
    x_deep_nobalance = x_list[1]
    x_deep_balance = x_list[2]
    y_meta = y_list[0]
    y_deep_nobalance = y_list[1]
    y_deep_balance = y_list[2]
    line, = plt.plot(x_meta, y_meta, marker='.', color="r", label="AdvMetaDet (ours)")
    line, = plt.plot(x_deep_nobalance, y_deep_nobalance, marker='.', color="b", label="DNN")
    line, = plt.plot(x_deep_balance, y_deep_balance, marker='.', color='y',
                     label="DNN(balanced)")
    plt.xlim(0, x_meta[-1] + 1)
    plt.ylim(30, 65)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.xticks([1,5,10,15,20,25,30,35,40,45,50], fontsize=15)
    plt.yticks([30,35,40,45,50,55,60,65], fontsize=15)
    plt.xlabel("fine-tune iterations", fontsize=18)
    plt.ylabel("F1 score of query set(%)", fontsize=18)
    plt.legend(loc='lower right', prop={'size': 17})
    plt.savefig(save_fig_name, dpi=200)

def draw_ways_line_curve(x, y, save_fig_name):
    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=(8, 6))
    line, = plt.plot(x, y, marker='.', color="r")
    plt.xlim(4, x[-1] + 1)
    plt.ylim(5, 75)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.xticks(x, fontsize=15)
    plt.yticks([5,10,15,20,25,30,35,40,45,50,55,60,65,70,75], fontsize=15)
    plt.xlabel("the number of way in each task", fontsize=18)
    plt.ylabel("F1 score of query set(%)", fontsize=18)
    plt.legend(loc='upper right', prop={'size': 17})
    plt.savefig(save_fig_name, dpi=200)


def draw_query_size_line_curve(x, y, save_fig_name):
    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=(8, 6))
    line, = plt.plot(x, y, marker='.', color="r")
    plt.xlim(4, x[-1] + 1)
    plt.ylim(59, 64)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.xticks(x, fontsize=15)
    plt.yticks([59,60,61,62,63,64], fontsize=15)
    plt.xlabel("query set size of each way during training", fontsize=18)
    plt.ylabel("F1 score of query set(%)", fontsize=18)
    plt.legend(loc='upper right', prop={'size': 17})
    plt.savefig(save_fig_name, dpi=200)

if __name__ == "__main__":
    # x, y  = read_json_data("/home1/machen/adv_detection_meta_learning/train_pytorch_model/fine_tune_update_ablation_study/fine_tune_update_ablation_study_result.json")
    # draw_finetune_line_curve(x,y, "/home1/machen/adv_detection_meta_learning/train_pytorch_model/fine_tune_update_ablation_study/fine_tune_ablation.png")

    # x_meta, y_meta = read_json_data(
    #     "/home1/machen/adv_detection_meta_learning/train_pytorch_model/fine_tune_update_ablation_study/fine_tune_update_ablation_study_result.json")
    # x_deep_nobalance, y_deep_nobalance, x_deep_balance, y_deep_balance = read_deep_learning_finetune_data(
    #     "/home1/machen/adv_detection_meta_learning/train_pytorch_model/DL_DET/evaluate_finetune_eval_using_TRAIN_I_TEST_II_protocol.json")
    # x_deep_nobalance = x_deep_nobalance[1:]
    # y_deep_nobalance = y_deep_nobalance[1:]
    # x_deep_balance = x_deep_balance[1:]
    # y_deep_balance = y_deep_balance[1:]
    # draw_finetune_line_curve([x_meta, x_deep_nobalance, x_deep_balance], [y_meta, y_deep_nobalance, y_deep_balance],
    #                          "/home1/machen/adv_detection_meta_learning/train_pytorch_model/fine_tune_update_ablation_study/finetune_ablation.png")


    x_meta, y_meta = read_json_data(
        "/home1/machen/adv_detection_meta_learning/train_pytorch_model/shots_ablation_study/shots_ablation_study_result.json")
    x_deep_nobalance, y_deep_nobalance, x_deep_balance, y_deep_balance = read_deep_learning_shots_data("/home1/machen/adv_detection_meta_learning/train_pytorch_model/DL_DET/evaluate_shots_eval_using_TRAIN_I_TEST_II_protocol.json")
    x_deep_nobalance = x_deep_nobalance[1:]
    y_deep_nobalance = y_deep_nobalance[1:]
    x_deep_balance = x_deep_balance[1:]
    y_deep_balance = y_deep_balance[1:]

    # shots需要将DeepLearning和MetaLearning都画上
    draw_shots_line_curve([x_meta, x_deep_nobalance, x_deep_balance], [y_meta, y_deep_nobalance, y_deep_balance],
                             "/home1/machen/adv_detection_meta_learning/train_pytorch_model/shots_ablation_study/shots_ablation.png")
    #

    # x,y = read_json_data("/home1/machen/adv_detection_meta_learning/train_pytorch_model/inner_update_ablation_study/inner_update_ablation_study_result.json")
    # draw_inner_update_line_curve(x,y,"/home1/machen/adv_detection_meta_learning/train_pytorch_model/inner_update_ablation_study/inner_update_ablation.png")
    # x, y = read_json_data(
    #     "/home1/machen/adv_detection_meta_learning/train_pytorch_model/tasks_ablation_study/tasks_ablation_study_result.json")
    # draw_tasks_line_curve(x, y,
    #                          "/home1/machen/adv_detection_meta_learning/train_pytorch_model/tasks_ablation_study/tasks_ablation.png")

    # x, y = read_json_data("/home1/machen/adv_detection_meta_learning/train_pytorch_model/query_size_ablation_study/query_size_ablation_study_result.json")
    # draw_query_size_line_curve(x,y, "/home1/machen/adv_detection_meta_learning/train_pytorch_model/query_size_ablation_study/query_size_ablation.png")