import json
import numpy as np
import math
import matplotlib.pyplot as plt
import periodicity_detection as period
import pandas as pd
import src.OnlineSLE.IIAI_synthetic as ut
from scipy.fft import fft, fftfreq
from src.utility.utility import result_aggregation
from src.OnlineSLE.OnlineSLE import OnlineSLE

results = []
# residual_rates = [0.01, 0.05, 0.1, 0.5, 0.75, 1]
residual_rates = [0.5]
for residual_rate in residual_rates:
    sliding_window_size = 500
    data = ut.generate_syn1(residual_rate = residual_rate, is_export = True)
    ts = np.array(data['ts'])
    answer = data['ground_truth']
    acf_result = []
    for idx in range(len(ts) - sliding_window_size + 1):
        W = ts[idx:sliding_window_size + idx]
        SLE_result = period.autocorrelation(W)

        acf_result.append({'idx_win': sliding_window_size + idx,
                           'answer': answer[sliding_window_size + idx - 1],
                           'residual_rate':residual_rate,
                           'result': SLE_result})

    results = results + result_aggregation('syn1', 'ACF', acf_result)

    autoperiod_result = []
    for idx in range(len(ts) - sliding_window_size + 1):
        W = ts[idx:sliding_window_size + idx]
        SLE_result = period.autoperiod(W)

        autoperiod_result.append({'idx_win': sliding_window_size + idx,
                           'answer': answer[sliding_window_size + idx - 1],
                           'residual_rate': residual_rate,
                           'result': SLE_result})

    results = results + result_aggregation('syn1', 'autoPERIOD', autoperiod_result)

    find_length_result = []
    for idx in range(len(ts) - sliding_window_size + 1):
        W = ts[idx:sliding_window_size + idx]
        SLE_result = period.find_length(W)

        find_length_result.append({'idx_win': sliding_window_size + idx,
                                  'answer': answer[sliding_window_size + idx - 1],
                                  'residual_rate': residual_rate,
                                  'result': SLE_result})

    results = results + result_aggregation('syn1', 'find_length', find_length_result)

    onlineSLE_result = []
    W = ts[:sliding_window_size]
    SLE = OnlineSLE(W, 'HAQSE')
    SLE_result = SLE.initial_phase()
    onlineSLE_result.append({'idx_win': sliding_window_size,
                             'answer': answer[sliding_window_size - 1],
                             'residual_rate': residual_rate,
                             'result': SLE_result})

    ## online mode
    for idx, x_t in enumerate(ts[sliding_window_size:]):
        SLE_result = SLE.online_phase(x_t)
        onlineSLE_result.append({'idx_win': 1 + sliding_window_size + idx,
                                 'answer': answer[sliding_window_size + idx],
                                 'residual_rate': residual_rate,
                                 'result': SLE_result})

    results = results + result_aggregation('syn1', 'Online_HAQSE', onlineSLE_result)

    ### syn2
    sliding_window_size = 400
    data = ut.generate_syn2(residual_rate=residual_rate, is_export=False)
    ts = np.array(data['ts'])
    answer = data['ground_truth']
    acf_result = []
    for idx in range(len(ts) - sliding_window_size + 1):
        W = ts[idx:sliding_window_size + idx]
        SLE_result = period.autocorrelation(W)

        acf_result.append({'idx_win': sliding_window_size + idx,
                           'answer': answer[sliding_window_size + idx - 1],
                           'residual_rate': residual_rate,
                           'result': SLE_result})

    results = results + result_aggregation('syn2', 'ACF', acf_result)

    autoperiod_result = []
    for idx in range(len(ts) - sliding_window_size + 1):
        W = ts[idx:sliding_window_size + idx]
        SLE_result = period.autoperiod(W)

        autoperiod_result.append({'idx_win': sliding_window_size + idx,
                                  'answer': answer[sliding_window_size + idx - 1],
                                  'residual_rate': residual_rate,
                                  'result': SLE_result})

    results = results + result_aggregation('syn2', 'autoPERIOD', autoperiod_result)

    find_length_result = []
    for idx in range(len(ts) - sliding_window_size + 1):
        W = ts[idx:sliding_window_size + idx]
        SLE_result = period.find_length(W)

        find_length_result.append({'idx_win': sliding_window_size + idx,
                                   'answer': answer[sliding_window_size + idx - 1],
                                   'residual_rate': residual_rate,
                                   'result': SLE_result})

    results = results + result_aggregation('syn2', 'find_length', find_length_result)

    onlineSLE_result = []
    W = ts[:sliding_window_size]
    SLE = OnlineSLE(W, 'HAQSE')
    SLE_result = SLE.initial_phase()
    onlineSLE_result.append({'idx_win': sliding_window_size,
                             'answer': answer[sliding_window_size - 1],
                             'residual_rate': residual_rate,
                             'result': SLE_result})

    ## online mode
    for idx, x_t in enumerate(ts[sliding_window_size:]):
        SLE_result = SLE.online_phase(x_t)
        onlineSLE_result.append({'idx_win': 1 + sliding_window_size + idx,
                                 'answer': answer[sliding_window_size + idx],
                                 'residual_rate': residual_rate,
                                 'result': SLE_result})

    results = results + result_aggregation('syn2', 'Online_HAQSE', onlineSLE_result)
    df = pd.DataFrame(results)
    # csv_file = 'output.csv'
    # df.to_csv(csv_file, index=False)