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


def draw_shots_line_curve(x, y, save_fig_name):
    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=(8, 6))
    line, = plt.plot(x, y, marker='.', color="r")
    plt.xlim(0, x[-1] + 1)
    plt.ylim(60, 90)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.xticks(x, fontsize=15)
    plt.yticks([60,65,70,75,80,85,90], fontsize=15)
    plt.xlabel("shot number", fontsize=18)
    plt.ylabel("F1 score of query set(%)", fontsize=18)
    plt.legend(loc='upper right', prop={'size': 17})
    plt.savefig(save_fig_name, dpi=200)


def draw_tasks_line_curve(x, y, save_fig_name):
    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=(8, 6))
    line, = plt.plot(x, y, marker='.', color="r")
    plt.xlim(4, x[-1] + 1)
    plt.ylim(60, 80)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.xticks(x, fontsize=15)
    plt.yticks([60,65,70,75,80], fontsize=15)
    plt.xlabel("task number", fontsize=18)
    plt.ylabel("F1 score of query set(%)", fontsize=18)
    plt.legend(loc='upper right', prop={'size': 17})
    plt.savefig(save_fig_name, dpi=200)

def draw_inner_update_line_curve(x, y, save_fig_name):
    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=(8, 6))
    line, = plt.plot(x, y, marker='.', color="r")
    plt.xlim(4, x[-1] + 1)
    plt.ylim(70, 80)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.xticks(x, fontsize=15)
    plt.yticks([70,72,74,76,78,80], fontsize=15)
    plt.xlabel("inner update times", fontsize=18)
    plt.ylabel("F1 score of query set(%)", fontsize=18)
    plt.legend(loc='upper right', prop={'size': 17})
    plt.savefig(save_fig_name, dpi=200)

def draw_finetune_line_curve(x, y, save_fig_name):
    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=(8, 6))
    line, = plt.plot(x[1:], y[1:], marker='.', color="r")
    plt.xlim(1, x[-1] + 1)
    plt.ylim(70, 76)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.xticks([0,5,10,15,20,25,30,35,40,45,50], fontsize=15)
    plt.yticks([70,72,74,76,78,80], fontsize=15)
    plt.xlabel("fine-tune iterations", fontsize=18)
    plt.ylabel("F1 score of query set(%)", fontsize=18)
    plt.legend(loc='upper right', prop={'size': 17})
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
    plt.ylim(70, 76)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.xticks(x, fontsize=15)
    plt.yticks([70,71,72,73,74,75,76], fontsize=15)
    plt.xlabel("the query set size in each way during training", fontsize=18)
    plt.ylabel("F1 score of query set(%)", fontsize=18)
    plt.legend(loc='upper right', prop={'size': 17})
    plt.savefig(save_fig_name, dpi=200)

if __name__ == "__main__":
    # x, y  = read_json_data("/home1/machen/adv_detection_meta_learning/train_pytorch_model/fine_tune_update_ablation_study/fine_tune_update_ablation_study_result.json")
    # draw_finetune_line_curve(x,y, "/home1/machen/adv_detection_meta_learning/train_pytorch_model/fine_tune_update_ablation_study/fine_tune_ablation.png")
    #
    # x, y = read_json_data(
    #     "/home1/machen/adv_detection_meta_learning/train_pytorch_model/inner_update_ablation_study/inner_update_ablation_study_result.json")
    # draw_inner_update_line_curve(x, y,
    #                          "/home1/machen/adv_detection_meta_learning/train_pytorch_model/inner_update_ablation_study/inner_update_ablation.png")
    #
    #
    # x, y = read_json_data(
    #     "/home1/machen/adv_detection_meta_learning/train_pytorch_model/shots_ablation_study/shots_ablation_study_result.json")
    # draw_shots_line_curve(x, y,
    #                          "/home1/machen/adv_detection_meta_learning/train_pytorch_model/shots_ablation_study/shots_ablation_study_result.png")
    #
    # x, y = read_json_data(
    #     "/home1/machen/adv_detection_meta_learning/train_pytorch_model/tasks_ablation_study/tasks_ablation_study_result.json")
    # draw_tasks_line_curve(x, y,
    #                          "/home1/machen/adv_detection_meta_learning/train_pytorch_model/tasks_ablation_study/tasks_ablation.png")

    x, y = read_json_data("/home1/machen/adv_detection_meta_learning/train_pytorch_model/query_size_ablation_study/query_size_ablation_study_result.json")
    draw_query_size_line_curve(x,y, "/home1/machen/adv_detection_meta_learning/train_pytorch_model/query_size_ablation_study/query_size_ablation.png")