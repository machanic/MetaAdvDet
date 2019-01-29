import random
import os


def genList(fileListPath, writePath, type="train"):
    fid = open(fileListPath, "r")
    wid = open(writePath, "w")
    root_dir = "/home1/machen/dataset/CIFAR-10/split_data_mem"
    for line in fid:
        if type not in line:
            continue
        line = line.strip()
        count_file_path = os.path.join(root_dir, os.path.dirname(line)) + '/count.txt'
        with open(count_file_path, "r") as file_obj:
            count = int(file_obj.read())
        arylines = line.split("/")
        label = int(arylines[1].split("_")[-1]) - 1
        for i in range(count):
            wid.write(line + "#{}".format(i) + " " + str(label) + "\n")
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


genList("/home1/machen/dataset/CIFAR-10/split_data_mem/train_image_list.txt",
        "/home1/machen/dataset/CIFAR-10/split_data_mem/train_image_label.txt", "train")

genList("/home1/machen/dataset/CIFAR-10/split_data_mem/test_image_list.txt",
        "/home1/machen/dataset/CIFAR-10/split_data_mem/test_image_label.txt", "test")
# genList("test_image_list.txt", "test_image_crop_support_list.txt", "Support")
# genList("test_image_list.txt", "test_image_crop_query_list.txt", "Query")
# genLists("test_image_list.txt", "test_image_crop_query_lists", "Query", 15, 100)
# genLists("test_image_list.txt", "test_image_crop_support_lists", "Support", 1, 100)
# genLists2("test_image_list.txt", "test_image_crop_support_lists", "test_image_crop_query_lists", 5, 15, 100)
