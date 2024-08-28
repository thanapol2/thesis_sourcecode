import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from matplotlib.animation import HTMLWriter, FuncAnimation
from scipy.fft import fftfreq
import sklearn.metrics as metric

def tricube_kernel(u: float):
    if 0 <= u < 1:
        return (1 - (u ** 3)) ** 3
    else:
        return 0


def symmetric_weights_bak(buffer_length: int, input_lambda: int):
    weights = np.zeros((buffer_length, buffer_length))

    for i in range(buffer_length):
        for j in range(buffer_length):
            u = abs((i - j) / input_lambda)
            weights[i][j] = tricube_kernel(u)

    return weights


# for symmetric_w ([x_{t-lambda/2},... x_{t-1}, x_t, x{t+1},...,x_{t+lambda/2})
# lambda must be window_size/2
# To fast computation, we collect weight in to [total_lenght x total_lenght]
# Total length of ini phase is 4m.
def symmetric_weights(total_length: int, buffer_size: int):
    weights = np.zeros((total_length, total_length))
    input_lambda = round(buffer_size / 2)
    # for each point, we compute weight with 2m

    for i in range(total_length):
        # only symmetric weight idx
        if i in range(input_lambda, total_length - input_lambda):
            for j in range(total_length):
                u = abs((i - j) / input_lambda)
                weights[i][j] = tricube_kernel(u)
        else:
            weights[i][:] = np.nan

    return weights


# window for each point <- [x_{t-lambda/2},... x_{t-1}, x_t, x{t+1},...,x_{t+lambda/2}
# Single seasonality m = m_p, then A is 4m and buffer for symmetric trend must be 2m
def symmetric_trend_filter(input_data: np.ndarray, buffer_size: int):
    total_length = len(input_data)
    weights = symmetric_weights(total_length, buffer_size)
    smoothed_trend = np.zeros(total_length)
    for i in range(total_length):
        local_weights = weights[i]
        smoothed_trend[i] = trend_filter(weights=local_weights, data=input_data)

    return smoothed_trend


def symmetric_trend_filter_bak(buffer: np.ndarray, input_lambda: int):
    buffer_length = len(buffer)
    weights = symmetric_weights_bak(buffer_length, input_lambda)
    smoothed_trend = np.zeros(buffer_length)
    for i in range(buffer_length):
        local_weights = weights[i]
        smoothed_trend[i] = trend_filter(weights=local_weights, data=buffer)

    return smoothed_trend


def non_symmetric_weights(input_lambda: int):
    weights = [tricube_kernel((input_lambda - k) / input_lambda) for k in range(1, input_lambda + 1)]
    return np.array(weights)


def trend_filter(weights: np.ndarray, data: np.ndarray):
    t_i = np.sum(data * weights) / np.linalg.norm(np.nan_to_num(weights), ord=1)
    return t_i


def seasonality_filter(d: float, Epsilon_r: float, gamma: float):
    de_Epsilon_r = gamma * d + (1 - gamma) * Epsilon_r
    return de_Epsilon_r



def update_array(x: np.ndarray, y):
    # Check if the array is empty
    if x.size == 0:
        return x  # If empty, nothing to update

    # Pop the oldest element (leftmost) from the deque
    x = x[1:]

    # Append the new value y to the deque
    x = np.append(x, y)

    return x

def simple_detread(data: np.ndarray):
    index = np.arange(data.shape[0])
    trend_fit = linregress(index, data)
    if trend_fit.slope > 1e-4:
        trend = trend_fit.intercept + index * trend_fit.slope
        data = data - trend
    else:
        data = data - data.mean()

    return data

def calculate_smoothness(ts):
    # Calculate the first-order difference
    first_order_difference = np.diff(ts)

    # Calculate the standard deviation of the first-order difference
    smoothness_measure = np.std(first_order_difference)

    return smoothness_measure

def calculate_rate_of_change(ts, seasonality_length):
    """
    Calculate the rate of change of the seasonal component (Evaluate CRAN dataset)
    """
    n = len(ts)
    num_cycles = -(-n // seasonality_length)
    padding_size = num_cycles * seasonality_length - n
    # Pad ts with None
    ts = np.concatenate([ts, np.full(padding_size, np.nan)])
    results = np.array([])
    for cycle_index in range(seasonality_length):
        subseasonality_ts = [ts[cycle_index + (i * seasonality_length)] for i in range(num_cycles)]
        subseasonality_ts = np.array(subseasonality_ts)
        subseasonality_ts = subseasonality_ts[~np.isnan(subseasonality_ts)]
        differences = abs(np.diff(subseasonality_ts))
        time_diff = np.diff(range(len(subseasonality_ts)))
        rate_of_change = differences / time_diff
        results = np.concatenate([results, rate_of_change])

    return np.mean(results), np.std(results)
