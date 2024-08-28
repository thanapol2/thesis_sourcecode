### Original source code from
# URL : https://github.com/thanapol2/onlineSLE/
# T. Phungtua-eng, Y. Yamamoto, 2024. A Fast Season Length Estimation Using Sliding Discrete Fourier Transform
# for Time Series Streaming Data. 16th International Congress on Advanced Applied Informatics (IIAI AAI 2024)


import json
import numpy as np
import math


def square_wave(length, period, amplitude):
    one_period = np.arange(0, period, 1)
    one_period = amplitude * np.sign(np.sin(2 * np.pi * (1 / period) * one_period))
    number_cycle = math.ceil(length / period)
    seasonal = []
    for i in range(number_cycle):
        seasonal = np.concatenate([seasonal, one_period])

    return seasonal[:length]


def sinewave(length: int, period: int, amplitude: int):
    one_period = np.arange(0, period, 1)
    frequency = 1 / period
    theta = 0
    one_period = amplitude * np.sin(2 * np.pi * frequency * one_period + theta)
    number_cycle = math.ceil(length / period)
    seasonal = []
    for i in range(number_cycle):
        seasonal = np.concatenate([seasonal, one_period])

    return seasonal[:length]


# All datasets are exported into JSON files
# Data dic :
# 1. ground_truth denotes the ground truth of the season length of each timestamp
# 2. ts denotes original time series data
# 3. seasonality denotes seasonality (S)
# 4. residual denotes residual (R)
def generate_syn1(filename: str = "syn1.json", residual_rate=0.1, is_export=False):
    # dataset oneShotSTL
    np.random.seed(0)
    M_P = 100
    TS_length = 5200

    seasonality = square_wave(TS_length, 100, 1)
    sigma = np.std(seasonality)
    residual = residual_rate * sigma * np.random.randn(TS_length)
    TS = seasonality + residual

    ground_truth = np.repeat(M_P, TS_length)
    data = {'ground_truth': ground_truth.tolist(),
            'ts': TS.tolist(),
            'seasonality': seasonality.tolist(),
            'residual': residual.tolist()}

    if is_export:
        with open(filename, "w") as outfile:
            json.dump(data, outfile)
    return data


def generate_syn2(filename: str = "syn2.json", residual_rate=0.1, is_export=False):
    np.random.seed(0)
    TS_length = 5000

    M_1 = 50
    M_2 = 80

    first_pattern = sinewave(1800, M_1, 1)
    second_pattern = sinewave(1800, M_2, 1)
    third_pattern = sinewave(1400, M_1, 1)
    seasonality = np.concatenate((first_pattern,
                                  second_pattern,
                                  third_pattern))

    ground_truth = np.concatenate((np.repeat(M_1, 1800),
                                   np.repeat(M_2, 1800),
                                   np.repeat(M_1, 1400)))

    sigma = np.std(seasonality)
    residual = residual_rate * sigma * np.random.randn(TS_length)
    TS = seasonality + residual

    data = {'ground_truth': ground_truth.tolist(),
            'ts': TS.tolist(),
            'seasonality': seasonality.tolist(),
            'residual': residual.tolist()}

    if is_export:
        with open(filename, "w") as outfile:
            json.dump(data, outfile)

    return data