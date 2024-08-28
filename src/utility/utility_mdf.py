import os
from astropy.time import Time
from datetime import datetime
# from utility.utility_webservice import MDFdataset_path, MDFdata_path

MDF_PATH = os.path.expanduser("~/lightcurve/MDF_data")
LC_PATH = "lc_flux_catalog_aperture_r{}_txt"
TIME_PATH = "timestamp_flux_catalog_aperture_r{}_txt"

# LC_PATH = os.path.join(MDF_PATH, "lc_flux_catalog_aperture_r7_txt")
# TIME_PATH = os.path.join(MDF_PATH, "timestamp_flux_catalog_aperture_r7_txt")

dataset_path = 'D:\\mdwarf_data\\'
lc_path = "lc_flux_catalog_aperture_r7_txt\\"
dy_path = "lc_flux_catalog_aperture_r7_dy\\"
lc_timestamp_path = "timestamp_flux_catalog_aperture_r7_txt\\"
path_to_lc_file = "{}{}".format(dataset_path, lc_path)
path_to_lc_timestamp = "{}{}".format(dataset_path, lc_timestamp_path)
path_to_dy_file = "{}{}".format(dataset_path, dy_path)
html_path = "lc_flux_catalog_aperture_r7_html\\"
png_path = "lc_flux_catalog_aperture_r7_png\\"

def get_lc_path(r):
    # lc_path_fixconfig = "{}lc_flux_catalog_aperture_r{}_txt\\".format(dataset_path,r)
    folderName = "lc_flux_catalog_aperture_r{}_txt".format(r)
    lc_path_fixconfig = os.path.join(MDF_PATH, folderName)
    return lc_path_fixconfig

def get_timestamp_path(r):
    folderName = "timestamp_flux_catalog_aperture_r{}_txt".format(dataset_path,r)
    lc_path_fixconfig = os.path.join(MDF_PATH, folderName)
    return lc_path_fixconfig


def get_data_from_file(file_name, r=7):
    lc_file = os.path.join(MDF_PATH, LC_PATH.format(r),"{}.txt".format(file_name))
    mjd_file = os.path.join(MDF_PATH, TIME_PATH.format(r),"{}.txt".format(file_name))
    with open(lc_file, 'r') as f:
        instances = [float(line.strip()) for line in f]

    with open(mjd_file, 'r') as f:
        timestamps = [float(line.strip()) for line in f]

    return {"file_name": file_name,
            "timestamps": timestamps,
            "instances": instances}


def get_list_MDF(r=7, startFile=None, pattern=None):
    indexList = None
    path_to_lc_file = os.path.join(MDF_PATH, LC_PATH.format(r))
    listFiles = []
    for r, d, files in os.walk(path_to_lc_file):
        for index, file in enumerate(files):
            pathFile = os.path.join(path_to_lc_file, file)
            filesize = os.path.getsize(pathFile)
            if filesize != 0:
                if pattern is not None:
                    if pattern in file:
                        listFiles.append(file.replace(".txt", ""))
                else:
                    listFiles.append(file.replace(".txt", ""))
        if startFile is not None:
            indexList = listFiles.index(startFile)
                # if "{}.txt".format(startFile) == file:
                #     indexList = index + 1
            if indexList is None:
                return listFiles
            else:
                return listFiles[indexList:]
        else:
            return listFiles


    # if indexList is None:
    #     return listFiles
    # else:
    #     return listFiles[indexList:]

def isOverlapTimestamp(timestamp1,timestamp2, uprange= 0.00003, lowrange=0.00003):
    start1_low = timestamp1[0] - lowrange
    start1_up = timestamp1[0] + uprange
    end1_low = timestamp1[-1] - lowrange
    end1_up = timestamp1[-1] + uprange

    start2_low = timestamp2[0] - lowrange
    start2_up = timestamp2[0] + uprange
    end2_low = timestamp2[-1] - lowrange
    end2_up = timestamp2[-1] + uprange

    check_start = (start1_low<=timestamp2[0]<=start1_up) & (start2_low<=timestamp1[0]<=start2_up)
    check_end = (end1_low <= timestamp2[-1] <= end1_up) & (end2_low <= timestamp1[-1] <= end2_up)

    # isOverlap = not ((timestamp1[-1]+uprange<timestamp2[0]-lowrange) or (timestamp2[-1]+uprange<timestamp1[0]-lowrange))
    isOverlap = check_start & check_end
    return isOverlap

def normalization_with_peaktime(mjd_list_timestamp:list[float], mjd_peak_time:float):
    UTC_peak_time = Time(mjd_peak_time, format='mjd').tt.datetime
    diff_timstamps = []
    for mjd_timestamp in mjd_list_timestamp:
        UTC_timestamp = Time(mjd_timestamp, format='mjd').tt.datetime
        diff_seconds = (UTC_timestamp - UTC_peak_time).total_seconds()
        diff_timstamps.append(diff_seconds)
    return diff_timstamps

