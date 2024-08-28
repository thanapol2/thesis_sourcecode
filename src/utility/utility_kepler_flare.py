import os
from random import uniform, randint
import numpy as np
from math import exp

KEPLER_PATH = os.path.expanduser("~/lightcurve/kepler_stable")
TOMOE_PATH = os.path.expanduser("~/lightcurve/tomoe_stable")


# re timestamp into timescale
# Eq 3 Aizawa
# time_FWHM (Full width at half maximum) = time_rise
def transform_timescale(time_input, time_peak, time_rise):
    time_scale = (time_input - time_peak) / time_rise
    return time_scale


# Kepler Flares paper II Eq.1 -1<t<=0
# Aizawa paper Eq.5
def kepler_rise_phase(time_scale, time_rise=1):
    # scale time t1/2
    t = time_scale
    a0 = 1
    a1 = (1.941 + uniform(-0.008, 0.008)) * t
    a2 = (-0.175 + uniform(-0.032, 0.032)) * t * t
    a3 = (-2.246 + uniform(-0.039, 0.039)) * t * t * t
    a4 = (-1.125 + uniform(-0.016, 0.016)) * t * t * t * t
    f_rise = a0 + a1 + a2 + a3 + a4
    return f_rise


# Kepler Flares paper II Eq.4 0<t < n
def kepler_decay_phase(time_scale):
    fast_phase = (0.6890 + uniform(- 0.0008, 0.0008)) * exp((-1.6 + uniform(-0.003, 0.003)) * time_scale)
    slow_phase = (0.3030 + uniform(- 0.0009, 0.0009)) * exp((-0.2783 + uniform(-0.0007, 0.0007)) * time_scale)
    return fast_phase + slow_phase


# Aizawa
def tomoe_rise_phase(time_scale):
    a0 = 1
    a1 = 4.54 * time_scale
    a2 = 8.31 * time_scale * time_scale
    a3 = 6.82 * time_scale * time_scale * time_scale
    a4 = 2.05 * time_scale * time_scale * time_scale * time_scale
    f_rise = a0 + a1 + a2 + a3 + a4
    return f_rise


def tomoe_peak_phase():
    return 1


def tomoe_decay_phase(time_scale):
    c = 0.85
    fast_phase = c * exp(-1.28 * time_scale)
    slow_phase = (1 - c) * exp(-0.080 * time_scale)
    return fast_phase + slow_phase


# Kepler Flares Aizawa paper Eq 5
def kepler_flare(time_rise=1, time_peak=0, time_decay=6, duration_rise=10, duration_decay=60):
    range_rise = time_rise / duration_rise
    timestamp_rise = np.arange(-time_rise, time_peak, range_rise).tolist()
    range_decay = time_decay / duration_decay
    timestamp_decay = np.arange(time_peak, time_decay, range_decay).tolist()
    timestamps_list = timestamp_rise + timestamp_decay
    flux_list = []
    for t in timestamps_list:
        time_scale = transform_timescale(time_input=t, time_peak=time_peak, time_rise=time_rise)
        # time_scale_list.append(time_scale)
        # rise phase
        if (-time_rise <= time_scale) & (time_scale <= time_peak):
            flux_list.append(kepler_rise_phase(time_scale=time_scale, time_rise=time_rise))
        # decay phase
        else:
            flux_list.append(kepler_decay_phase(time_scale=time_scale))

    return timestamps_list, flux_list


# Aizawa Eq 4
# TIC ID = '55288759' shortest of time rise in aizawa paper 5 sec, delta 2 sec
def tomoe_flare(time_rise=1, time_peak=0, delta_timepeak=0.23, time_decay=6, duration_rise=10, duration_decay=60):
    range_rise = time_rise / duration_rise
    timestamp_rise = np.arange(-time_rise, time_peak, range_rise).tolist()
    timestamp_peak = np.arange(time_peak, time_peak + delta_timepeak, 0.05).tolist()
    range_decay = time_decay / duration_decay
    timestamp_decay = np.arange(time_peak + delta_timepeak, time_decay, range_decay).tolist()
    timestamps_list = timestamp_rise + timestamp_peak + timestamp_decay

    flux_list = []
    for t in timestamps_list:
        time_scale = transform_timescale(time_input=t, time_peak=time_peak, time_rise=time_rise)
        # rise phase
        if time_scale <= time_peak:
            flux_list.append(tomoe_rise_phase(time_scale=time_scale))
        # peak phase
        elif (time_peak < time_scale) & (time_scale <= time_peak + delta_timepeak):
            flux_list.append(tomoe_peak_phase())
        # decay phase
        else:
            flux_list.append(tomoe_decay_phase(time_scale=time_scale))

    return timestamps_list, flux_list


def get_flare_data(flare_type, file_name):
    instances = []
    if flare_type == 'kepler':
        file_target = os.path.join(KEPLER_PATH, file_name + ".txt")
    elif flare_type == 'tomoe':
        file_target = os.path.join(TOMOE_PATH, file_name + ".txt")
    with open(file_target, 'r') as f:
        next(f)
        temp_bin = []
        for i, line in enumerate(f):
            instance = float(line.strip())
            instances.append(instance)
    information_flare = file_name.split("_")
    flare_height = information_flare[1][1:]
    flare_duration = information_flare[2][1:]
    time_start = information_flare[3]
    time_peak = information_flare[4]
    timestamp = [*range(0, len(instances))]
    return {
        'file_name': file_name,
        'instances': instances,
        'timestamps': timestamp,
        'flare_type': flare_type,
        'flare_height': int(flare_height),
        'flare_duration': int(flare_duration),
        'time_start': int(time_start),
        'time_peak': int(time_peak)

    }

def default_kepler_flare(peak_height):
    time_rise = 1
    time_decay = 6
    duration_rise = 10
    duration_decay = 110
    flare_timestamps, flare_flux = kepler_flare(time_rise=time_rise, time_decay=time_decay,
                                                duration_rise=duration_rise,
                                                duration_decay=duration_decay)
    flare_model = np.array(flare_flux) * peak_height
    return flare_model


def get_files_list(flare_type = 'kepler'):
    if flare_type == 'kepler':
        file_list = os.listdir(KEPLER_PATH)
    elif flare_type == 'tomoe':
        file_list = os.listdir(TOMOE_PATH)
    else:
        return None
    # Iterate over the files
    return_list = []
    for file_name in file_list:
        # Check if the file is a text file
        if not file_name.startswith(".DS_Store"):
            if file_name.endswith(".txt"):
                # Remove the ".txt" extension from the file name
                return_list.append(file_name[:-4])
            else:
                return_list.append(file_name)
    return return_list
