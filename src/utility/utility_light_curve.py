from tslearn.preprocessing import TimeSeriesScalerMeanVariance
import os
from math import sqrt
import pandas as pd
import fnmatch

# MDFdataset_path = 'C:\\git\\data_stream\\lightcurve_benchmark\\pca\\'
MDF_PATH = os.path.expanduser("~/lightcurve/pca")
WAVELET_PATH = os.path.expanduser("~/lightcurve/wavelet")
TRAINING_FOLDER = 'training'


def getFullPath(height, duration, pattern='sq'):
    folderName = "{}_L{}_I{}\\".format(pattern, height, duration)
    path_to_lc_file = os.path.join(MDF_PATH, folderName)
    # path_to_lc_file = "{}{}".format(MDFdataset_path, folderName)
    return path_to_lc_file


# stop use getListLight
def getListLight(height, duration, pattern='sq', folderType="test", startFile=None, isFullPath=False):
    indexList = None
    folderName = "{}_L{}_I{}".format(pattern, height, duration)
    path_to_lc_file = os.path.join(MDF_PATH, folderName, folderType)
    # path_to_lc_file = "{}{}{}\\".format(MDFdataset_path, folderName, folderType)
    listFiles = []
    for r, d, files in os.walk(path_to_lc_file):
        for index, file in enumerate(files):
            pathFile = os.path.join(path_to_lc_file, file)
            filesize = os.path.getsize(pathFile)
            if filesize != 0:
                if isFullPath:
                    listFiles.append(os.path.join(path_to_lc_file, file.replace(".txt", "")))
                else:
                    listFiles.append(file.replace(".txt", ""))
            if startFile is not None:
                if "{}.txt".format(startFile) == file:
                    indexList = index + 1

    if indexList is None:
        return listFiles
    else:
        return listFiles[indexList:]


def genListAns(start, duration):
    return [*range(int(start), int(start) + int(duration), 1)]


def load_allFileToList(input_path):
    listData = []
    listFileName = []
    for r, d, files in os.walk(input_path):
        # self.name_list.append(self._testfile)
        for file in files:
            fullName = os.path.join(r, file)
            listFileName.append(fullName)

            test_list = []
            # print("#### start file {}    ###########".format(test_file))
            with open(fullName) as txt_lines:
                for line in txt_lines:
                    test_list.append(int(line.replace('\n', '')))

            listData.append(test_list)
    return listFileName


def gen_list_mix(main_path, data_type, mix_size="*"):
    title_list = []
    pattern = "Mix*"
    directorys = os.listdir("{}{}".format(main_path, data_type))
    directorys_matching = fnmatch.filter(directorys, pattern)
    directorys_matching = fnmatch.filter(directorys_matching, mix_size)
    for index, directory in enumerate(directorys_matching):
        directorys_matching[index] = "{}_{}".format(data_type, directory)
    return directorys_matching


def gen_list_title(data_type):
    title_list = []
    LS = [3, 5]
    IS = [5, 10, 30, 60]
    patterns = ["sq"]
    for pattern in patterns:
        for L in LS:
            for I in IS:
                title_list.append("{}_{}_L{}_I{}".format(data_type, pattern, L, I))
    return title_list




def deconvert_databin_dynamic(instances):
    data_after_bin = []
    for index, instance in enumerate(instances):
        n = instance.get_number_instance()
        mean = instance.get_representation()
        for i in range(n):
            data_after_bin.append(mean)

    return data_after_bin


def cal_TF(change_points, tran_list, size_instances, bin_size=3):
    total_alert = len(change_points)
    found = False
    true_alerts = len(set(change_points) & set(tran_list))
    if true_alerts > 0:
        found = True
    false_alerts = total_alert - true_alerts
    pos = len(tran_list)
    size_neg = gen_Timestamp_Tran_list(start=0, end=size_instances, bin_size=bin_size)
    neg = len(size_neg) - len(tran_list)
    tp_rate = float(true_alerts) / pos
    fp_rate = float(false_alerts) / neg
    if true_alerts + false_alerts == 0:
        precision = 0
    else:
        precision = float(true_alerts) / float(true_alerts + false_alerts)
    if tp_rate + precision == 0:
        f1_measurement = 0
    else:
        f1_measurement = float(2) * (tp_rate * precision) / (tp_rate + precision)

    row = [true_alerts,
           false_alerts,
           total_alert,
           pos,
           neg,
           tp_rate, fp_rate,
           precision, f1_measurement,
           found]

    return row


def cal_TF_NoBin(change_points, tran_list, size_instances):
    total_alert = len(change_points)
    found = False
    true_alerts = len(set(change_points) & set(tran_list))
    if true_alerts > 0:
        found = True
    false_alerts = total_alert - true_alerts
    pos = len(tran_list)
    neg = size_instances - len(tran_list)
    tp_rate = float(true_alerts) / pos
    fp_rate = float(false_alerts) / neg
    if true_alerts + false_alerts == 0:
        precision = 0
    else:
        precision = float(true_alerts) / float(true_alerts + false_alerts)
    if tp_rate + precision == 0:
        f1_measurement = 0
    else:
        f1_measurement = float(2) * (tp_rate * precision) / (tp_rate + precision)

    row = [true_alerts,
           false_alerts,
           total_alert,
           pos,
           neg,
           tp_rate, fp_rate,
           precision, f1_measurement,
           found]

    return row


def cal_TF_NoBin_MixTran(change_points, tran_list, size_instances):
    total_alert = len(change_points)
    true_alerts = 0
    found = 0
    pos = 0
    for tran in tran_list:
        true_alert = len(set(change_points) & set(tran))
        true_alerts = true_alerts + true_alert
        if true_alert > 0:
            found = found + 1
        pos = pos + len(tran)

    false_alerts = total_alert - true_alerts
    neg = size_instances - pos
    tp_rate = float(true_alerts) / pos
    fp_rate = float(false_alerts) / neg
    if true_alerts + false_alerts == 0:
        precision = 0
    else:
        precision = float(true_alerts) / float(true_alerts + false_alerts)
    if tp_rate + precision == 0:
        f1_measurement = 0
    else:
        f1_measurement = float(2) * (tp_rate * precision) / (tp_rate + precision)

    row = [true_alerts,
           false_alerts,
           total_alert,
           pos,
           neg,
           tp_rate, fp_rate,
           precision, f1_measurement,
           found]

    return row


# row = [true_alerts,
#           false_alerts,
#           total_alert,
#           pos,
#           neg,
#           tp_rate,fp_rate,
#           precision,f1_measurement,
#           found]
def calRowTF_Bin(start, end, change_points, size_instances, bin_size=3):
    tran_list = gen_Timestamp_Tran_list(start=start, end=end, bin_size=bin_size)
    row = cal_TF(change_points=change_points,
                 tran_list=tran_list,
                 bin_size=bin_size,
                 size_instances=size_instances)
    return row


# row = [true_alerts,
#           false_alerts,
#           total_alert,
#           pos,
#           neg,
#           tp_rate,fp_rate,
#           precision,f1_measurement,
#           found]
def calRowTF_NoBin(start, end, change_points, size_instances, bin_size):
    tran_list = range(start, end)
    change_points = gen_change_points_list(change_points=change_points, bin_size=bin_size)
    row = cal_TF_NoBin(change_points=change_points,
                       tran_list=tran_list,
                       size_instances=size_instances)
    return row


# row = [true_alerts,
#           false_alerts,
#           total_alert,
#           pos,
#           neg,
#           tp_rate,fp_rate,
#           precision,f1_measurement,
#           found]
def calRowTF_NoBin_MixTran(st_ed_list, change_points, size_instances, bin_size):
    tran_list = []
    for st_ed in st_ed_list:
        tran_list.append(range(st_ed[0], st_ed[1]))
    change_points = gen_change_points_list(change_points=change_points, bin_size=bin_size)
    row = cal_TF_NoBin_MixTran(change_points=change_points,
                               tran_list=tran_list,
                               size_instances=size_instances)
    return row


def gen_change_points_list(change_points, bin_size):
    new_change_point = []
    for index, change_point in enumerate(change_points):
        i = 0
        while i < bin_size:
            new_change_point.append(change_point - i)
            i = i + 1
    return new_change_point


def gen_Timestamp_Tran_list(start, end, bin_size):
    timestamp_list = []
    full_list = range(start, end)
    for i in full_list:
        if (i % bin_size == 0) & (i - 1 >= start):
            timestamp_list.append(i - 1)
    return timestamp_list


def last_df_list(head_list):
    name_list = gen_list_title(data_type='bgs')
    return_list = []
    for name in name_list:
        return_list.append([name,
                            9999,
                            "999999",
                            "999999",
                            "999999",
                            "999999",
                            "999999",
                            1.0, 1.0,
                            "476"])

    return_df = pd.DataFrame(return_list, columns=head_list)
    return return_df


def export_list_bin_txt(max_num=[], max_bin=[], txt_file='temp.txt'):
    with open(txt_file, 'a') as output:
        output.write("max instance of bin = {}".format(max_num) + '\n')
        for row in max_bin:
            output.write(str(row) + '\n')


def calTtestTwoBin(priorBin, currentBin):
    priorMean = priorBin.get_representation()
    priorLower = priorBin.get_varianceDivNumber()

    currentMean = currentBin.get_representation()
    currentLower = currentBin.get_varianceDivNumber()

    Ttest = abs(priorMean - currentMean) / sqrt(priorLower + currentLower)

    return Ttest


def calFtestTwoBin(priorBin, currentBin):
    priorVar = priorBin.get_max_variance()
    currentVar = currentBin.get_max_variance()

    if priorVar == 0.0 or currentVar == 0.0:
        Ftest = 0.0
    else:
        Ftest = priorVar / currentVar

    return Ftest


def calTtestTwoBinWithVarianceWindow(priorBin, currentBin, varianceWindow):
    priorMean = priorBin.get_representation()
    priorLower = float(varianceWindow / priorBin.get_number_instance())

    currentMean = currentBin.get_representation()
    currentLower = float(varianceWindow / currentBin.get_number_instance())

    Ttest = abs(priorMean - currentMean) / sqrt(priorLower + currentLower)

    return Ttest


def get_list_file(height, duration, pattern='sq', startFile=None):
    folderName = "{}_L{}_I{}".format(pattern, height, duration)
    # path_to_lc_file = "{}{}".format(MDFdataset_path, folderName)
    path_to_lc_file = os.path.join(MDF_PATH, folderName, 'test')
    indexList = None
    listFiles = []
    for r, d, files in os.walk(path_to_lc_file):
        for index, file in enumerate(files):
            pathFile = os.path.join(path_to_lc_file, file)
            filesize = os.path.getsize(pathFile)
            if filesize != 0:
                listFiles.append(file.replace(".txt", ""))
            if startFile is not None:
                if "{}.txt".format(startFile) == file:
                    indexList = index + 1

    if indexList is None:
        return listFiles
    else:
        return listFiles[indexList:]


def get_data_from_file(file_name, height, duration, pattern, is_normalization=False):
    data_nor = None
    folder_name = "{}_L{}_I{}".format(pattern, height, duration)
    path_to_lc_file = os.path.join(MDF_PATH, folder_name)
    file_target = os.path.join(path_to_lc_file, 'test', file_name)
    file_answer = os.path.join(path_to_lc_file, 'answer', file_name)
    with open(file_target, 'r') as f:
        instances = [float(line.strip()) for line in f]
    if is_normalization:
        data_nor = TimeSeriesScalerMeanVariance(mu=0., std=1.).fit_transform([instances])
        data_nor = list(data_nor[0].ravel())
    timestamp = [*range(0, len(instances))]
    ansList = []
    if int(height) != 0:
        with open(file_answer, 'r') as f:
            ansIndexs  = [float(line.strip()) for line in f]
        for ans in ansIndexs:
            startIndex = int(ans)
            endIndex = int(ans) + int(duration)
            ansList = ansList + [*range(startIndex, endIndex)]
    return {"fileName": file_name,
            "height": height,
            "duration": duration,
            "timestamps": timestamp,
            "instances": instances,
            "instances_nor": data_nor,
            "ansList": ansList}


def get_list_LCfolder():
    heights = [1, 3]
    durations = [60, 100, 200, 500]
    patterns = ['sq', 'tr']
    list_folder_eva = []

    for pattern in patterns:
        for height in heights:
            for duration in durations:
                folder_name = f"{pattern}_I{height}_L{duration}"
                list_folder_eva.append({
                    "pattern": pattern,
                    "height": height,
                    "duration": duration,
                    "folder_name": folder_name
                })

    return list_folder_eva

def get_list_training_file():
    list_files = []
    path_to_lc_file = os.path.join(MDF_PATH, TRAINING_FOLDER)
    for r, d, files in os.walk(path_to_lc_file):
        for index, file in enumerate(files):
            path_file = os.path.join(path_to_lc_file, file)
            filesize = os.path.getsize(path_file)
            if filesize != 0:
                list_files.append(file.replace(".txt", ""))
    return list_files

def get_data_training_from_file(file_name):
    path_to_lc_file = os.path.join(MDF_PATH, TRAINING_FOLDER)
    file_target = os.path.join(path_to_lc_file, file_name)
    instances = []
    with open(file_target, 'r') as f:
        next(f)
        temp_bin = []
        for i, line in enumerate(f):
            instance = float(line.strip())
            instances.append(instance)
        # instances = [float(line.strip()) for line in f]
    timestamp = [*range(0, len(instances))]

    return {"file_name": file_name,
            "timestamps": timestamp,
            "instances": instances}