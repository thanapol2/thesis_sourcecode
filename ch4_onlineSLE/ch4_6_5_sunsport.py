import json
import numpy as np
import periodicity_detection as period
import pandas as pd
from scipy.fft import fft, fftfreq
from src.OnlineSLE.OnlineSLE import OnlineSLE
import os
from tqdm import tqdm

file_name = os.path.expanduser("~/source_code/thesis_sourcecode/datasets/06_sunspots.json")
with open(file_name, 'r') as file:
    json_data = json.load(file)

sliding_window_sizes = [660]
answers = [132]
err_bounds = [range(105, 160)]
results = []

for answer, err_bound, sliding_window_size in zip( answers, err_bounds, sliding_window_sizes):
    ts_data = np.array(json_data['ts'])
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
print(df)
csv_file = 'sunspot.csv'
df.to_csv(csv_file, index=False)
