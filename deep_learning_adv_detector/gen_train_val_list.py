import random
import os
import re

def genList(fileListPath, writePath, split_type="I", type="train"):
    fid = open(fileListPath, "r")
    wid = open(writePath, "w")
    root_dir = "/home1/machen/dataset/CIFAR-10/split_data_mem"
    for line in fid:
        if type not in line or split_type not in line:
            continue
        line = line.strip()
        count_file_path = os.path.join(root_dir, os.path.dirname(line)) + '/count.txt'
        with open(count_file_path, "r") as file_obj:
            count = int(file_obj.read())
        arylines = line.split("/")
        label = int(arylines[2].split("_")[-1]) - 1
        if label != 0:
            label = 1
        for i in range(count):
            wid.write(line + "#{}".format(i) + " " + str(label) + "\n")
    wid.flush()
    wid.close()


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


def hasPositiveSampleKey(keys):
    for key in keys:
        if key.split("_")[2] == "1":
            return True
    return False


def genLists(fileListPath, writePath, type, numSamples, numList):
    if not os.path.exists(writePath):
        os.mkdir(writePath)

    fid = open(fileListPath)
    labelMap = {}
    for line in fid:
        if type not in line:
            continue
        line = line.strip()
        arylines = line.split("/")
        label = arylines[1]
        if label not in labelMap:
            labelMap[label] = []
            labelMap[label].append(line)
        else:
            labelMap[label].append(line)

    for i in range(numList):
        wid = open(writePath + "/" + writePath + str(i) + ".txt", "w")
        labelMapSampled = random.sample(labelMap.keys(), 5)
        while True:
            if hasPositiveSampleKey(labelMapSampled) == True:
                break
            else:
                labelMapSampled = random.sample(labelMap.keys(), 5)

        for label in labelMapSampled:
            tempList = labelMap[label]
            for j in range(numSamples):
                imagePath = labelMap[label][j]
                if numSamples != len(tempList):
                    index = random.randint(0, len(tempList) - 1)
                    imagePath = labelMap[label][index]
                labelStr = "0"
                if label.split("_")[2] != "1":
                    labelStr = "1"
                wid.write(imagePath + " " + labelStr + "\n")
        wid.close()


def genLists2(fileListPath, support_writePath, query_writePath, support_numSamples, query_numSamples, numList):
    if not os.path.exists(support_writePath):
        os.mkdir(support_writePath)
    if not os.path.exists(query_writePath):
        os.mkdir(query_writePath)

    fid = open(fileListPath)
    labelMap = {}
    labelMap_query = {}
    for line in fid:
        line = line.strip()
        arylines = line.split("/")
        label = arylines[1]
        if "Support" in line:
            if label not in labelMap:
                labelMap[label] = []
            labelMap[label].append(line)
        else:
            if label not in labelMap_query:
                labelMap_query[label] = []
            labelMap_query[label].append(line)

    for i in range(numList):
        wid = open(support_writePath + "/" + support_writePath + str(i) + ".txt", "w")
        wid2 = open(query_writePath + "/" + query_writePath + str(i) + ".txt", "w")
        labelMapSampled = random.sample(labelMap.keys(), 5)
        while True:
            if hasPositiveSampleKey(labelMapSampled) == True:
                break
            else:
                labelMapSampled = random.sample(labelMap.keys(), 5)

        for label in labelMapSampled:
            tempList = labelMap[label]
            for j in range(support_numSamples):
                imagePath = labelMap[label][j]
                if support_numSamples != len(tempList):
                    index = random.randint(0, len(tempList) - 1)
                    imagePath = labelMap[label][index]
                labelStr = "0"
                if label.split("_")[2] != "1":
                    labelStr = "1"
                wid.write(imagePath + " " + labelStr + "\n")
        wid.close()

        for label in labelMapSampled:
            tempList = labelMap_query[label]
            for j in range(query_numSamples):
                imagePath = labelMap_query[label][j]
                if query_numSamples != len(tempList):
                    index = random.randint(0, len(tempList) - 1)
                    imagePath = labelMap_query[label][index]
                labelStr = "0"
                if label.split("_")[2] != "1":
                    labelStr = "1"
                wid2.write(imagePath + " " + labelStr + "\n")
        wid2.close()

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
