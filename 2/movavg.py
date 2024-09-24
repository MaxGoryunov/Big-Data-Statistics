import numpy as np
import pandas as pd
from scipy.stats import kendalltau

h = 0.05
k = np.arange(501)
noise = np.random.normal(0, 1, 501)
x_k = np.sqrt(k * h) + noise
df = pd.DataFrame(x_k, columns=['x_k'])


# Function to calculate moving averages and medians
def calculate_moving_average(x: np.ndarray, sizes):
    trends = {}
    for window in sizes:
        trend = np.array([0] * len(x))
        trend[0] = x[0]
        trend[-1] = x[-1]
        gap = window // 2
        for i in range(3, gap):
            trend[i] = np.average(x[0:2 * i + 1])
        for i in range(gap, 500 - gap + 1):
            trend[i] = np.average(x[i - gap:i + gap + 1])
        for i in range(3, gap):
            trend[500 - i] = np.average(x[500 - 2 * i - 1:])
        trends[f"average_{window}"] = trend
    return trends


def calculate_moving_median(x: np.ndarray, sizes):
    trends = {}
    for window in sizes:
        trend = np.array([0] * len(x))
        trend[0] = x[0]
        trend[-1] = x[-1]
        gap = window // 2
        for i in range(3, gap):
            trend[i] = np.median(x[0:2 * i + 1])
        for i in range(gap, 500 - gap + 1):
            trend[i] = np.average(x[i - gap:i + gap + 1])
        for i in range(3, gap):
            trend[500 - i] = np.average(x[500 - 2 * i - 1:])
        trends[f"median_{window}"] = trend
    return trends


def turning_points(x: np.ndarray):
    n = len(x)
    count = 0
    for i in range(0, n - 3):
        if (x[i] < x[i + 1]) and (x[i + 1] > x[i + 2]) \
                or (x[i] > x[i + 1]) and (x[i + 1] < x[i + 2]):
            count += 1
    return count, 2 * (n - 2) / 3, (16 * n - 29) / 90


def kendall_rank_coefficient(x: np.ndarray):
    p = 0
    n = len(x)
    for i in range(0, n - 1):
        for j in range(i + 1, n):
            if x[i] < x[j]:
                p += 1
    return 4 * p / (n * (n - 1)) - 1, 0, 2 * (2 * n + 5) / (9 * n * (n - 1))


windows = [21, 51, 111]
trends = calculate_moving_average(x_k, windows) | calculate_moving_median(x_k, windows)
results = {}

for key, trend in trends.items():
    residuals = df['x_k'] - trend
    p, Ep, Dp = turning_points(residuals)
    print(p, Ep, Dp)
    if (Ep - 3 * Dp < p) and (p <= Ep + 3 * Dp):
        print(f"По числу поворотных точек ряд {key} случаен")
    else:
        print(f"По числу поворотных точек ряд {key} не случаен")
    t, Et, Dt = kendall_rank_coefficient(residuals)
    print(t, Et, Dt)
    if (Et - 3 * Dt < t) and (t <= Et + 3 * Dt):
        print(f"По коэффициенту Кендела ряд {key} случаен")
    else:
        print(f"По коэффициенту Кендела ряд {key} не случаен")