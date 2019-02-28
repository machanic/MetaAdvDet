import random
import os
import re


def genTaskList(fileListPath, writePath, split_type="I", type="train"):
    fid = open(fileListPath, "r")
    wid = open(writePath, "w")
    pattern = re.compile(".*?/\d+_(\d+)/.*")
    for line in fid:
        if type not in line or split_type not in line:
            continue
        line = line.strip()
        npy_path = line.split()[-1]
        ma = pattern.match(npy_path)
        label = int(ma.group(1))
        if label != 1:  # clean is labeled as 1
            label = 0
        wid.write("{0} {1}\n".format(npy_path, label))
    wid.flush()
    wid.close()

#
# genTaskList("/home1/machen/adv_detection_meta_learning/task/train.txt",
#         "/home1/machen/dataset/CIFAR-10/split_data_mem/train_image_label.txt","/I/", "train")
# genTaskList("/home1/machen/adv_detection_meta_learning/task/test.txt",
#         "/home1/machen/dataset/CIFAR-10/split_data_mem/test_image_label.txt", "/II/", "test")


genTaskList("/home1/machen/adv_detection_meta_learning/task/TRAIN_I_TEST_II/train_CIFAR-10_tot_num_tasks_20000_metabatch_5_way_5_shot_5_query_15.txt",
        "/home1/machen/dataset/CIFAR-10/split_data_mem/TRAIN_I_TEST_II_train_CIFAR-10_tot_num_tasks_20000_metabatch_5_way_5_shot_5_query_15.txt","/I/", "train")
genTaskList("/home1/machen/adv_detection_meta_learning/task/TRAIN_I_TEST_II/test_CIFAR-10_tot_num_tasks_20000_metabatch_5_way_5_shot_5_query_15.txt",
        "/home1/machen/dataset/CIFAR-10/split_data_mem/TRAIN_I_TEST_II_test_CIFAR-10_tot_num_tasks_20000_metabatch_5_way_5_shot_5_query_15.txt", "/II/", "test")

genTaskList("/home1/machen/adv_detection_meta_learning/task/TRAIN_I_TEST_II/train_CIFAR-10_tot_num_tasks_20000_metabatch_5_way_5_shot_1_query_15.txt",
        "/home1/machen/dataset/CIFAR-10/split_data_mem/TRAIN_I_TEST_II_train_CIFAR-10_tot_num_tasks_20000_metabatch_5_way_5_shot_1_query_15.txt","/I/", "train")
genTaskList("/home1/machen/adv_detection_meta_learning/task/TRAIN_I_TEST_II/test_CIFAR-10_tot_num_tasks_20000_metabatch_5_way_5_shot_1_query_15.txt",
        "/home1/machen/dataset/CIFAR-10/split_data_mem/TRAIN_I_TEST_II_test_CIFAR-10_tot_num_tasks_20000_metabatch_5_way_5_shot_1_query_15.txt", "/II/", "test")

genTaskList("/home1/machen/adv_detection_meta_learning/task/TRAIN_I_TEST_II/train_CIFAR-10_tot_num_tasks_40000_metabatch_10_way_5_shot_5_query_30.txt",
        "/home1/machen/dataset/CIFAR-10/split_data_mem/TRAIN_I_TEST_II_train_CIFAR-10_tot_num_tasks_40000_metabatch_10_way_5_shot_5_query_30.txt","/I/", "train")
genTaskList("/home1/machen/adv_detection_meta_learning/task/TRAIN_I_TEST_II/test_CIFAR-10_tot_num_tasks_40000_metabatch_10_way_5_shot_5_query_30.txt",
        "/home1/machen/dataset/CIFAR-10/split_data_mem/TRAIN_I_TEST_II_test_CIFAR-10_tot_num_tasks_40000_metabatch_10_way_5_shot_5_query_30.txt", "/II/", "test")


# genList("test_image_list.txt", "test_image_crop_support_list.txt", "Support")
# genList("test_image_list.txt", "test_image_crop_query_list.txt", "Query")
# genLists("test_image_list.txt", "test_image_crop_query_lists", "Query", 15, 100)
# genLists("test_image_list.txt", "test_image_crop_support_lists", "Support", 1, 100)
# genLists2("test_image_list.txt", "test_image_crop_support_lists", "test_image_crop_query_lists", 5, 15, 100)
