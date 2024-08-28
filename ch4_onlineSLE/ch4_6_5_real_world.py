import json
import numpy as np
from scipy.stats import zscore
import periodicity_detection as period
import pandas as pd
from src.OnlineSLE.OnlineSLE import OnlineSLE
import os
from tqdm import tqdm

results = []
file_names = ['TiltECG_200_25000', 'TiltABP_210_25000']

sliding_window_sizes = [1000, 1050]
answers = [200, 210]
err_bounds = [range(160, 240),range(158, 242)]

for file_name, answer, err_bound, sliding_window_size in zip(file_names, answers, err_bounds, sliding_window_sizes):
    full_path = os.path.expanduser(f'~/source_code/thesis_sourcecode/datasets/{file_name}.txt')
    ts_data = np.loadtxt(full_path)
    ts_data = zscore(ts_data)
    best_match = 0
    err_match = 0
    count = 0
    for idx in tqdm(range(len(ts_data) - sliding_window_size + 1)):
        W = ts_data[idx:sliding_window_size + idx]
        SLE_result = period.find_length(W)
        if SLE_result == answer:
            best_match += 1
        if SLE_result in err_bound:
            err_match += 1
        count += 1

    results.append({'algo':'findlength',
                    'dataset': file_name,
                    'best_match': best_match/count,
                    '20_bound': err_match/count})

    best_match = 0
    err_match = 0
    count = 0
    for idx in tqdm(range(len(ts_data) - sliding_window_size + 1)):
        W = ts_data[idx:sliding_window_size + idx]
        SLE_result = period.autoperiod(W)
        if SLE_result == answer:
            best_match += 1
        if SLE_result in err_bound:
            err_match += 1
        count += 1

    results.append({'algo': 'autoperiod',
                    'dataset': file_name,
                    'best_match': best_match / count,
                    '20_bound': err_match / count})

    best_match = 0
    err_match = 0
    count = 0
    W = ts_data[:sliding_window_size]
    SLE = OnlineSLE(W, None)
    SLE_result = SLE.initial_phase()

    if SLE_result == answer:
        best_match += 1
    if SLE_result in err_bound:
        err_match += 1
    count += 1

    ## online mode
    for idx, x_t in enumerate(ts_data[sliding_window_size:]):
        SLE_result = SLE.online_phase(x_t)
        if SLE_result == answer:
            best_match += 1
        if SLE_result in err_bound:
            err_match += 1
        count += 1

    results.append({'algo':'sdft',
                    'dataset': file_name,
                    'best_match': best_match/count,
                    '20_bound': err_match/count})

df = pd.DataFrame(results)
csv_file = 'real_world_z.csv'
df.to_csv(csv_file, index=False)
