import math

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

# Step 1: Generate the data
# np.random.seed(42)  # For reproducibility
h = 0.05
k_values = np.arange(501)
normal_random_variables = np.random.normal(0, 1, 501)
x_k = np.sqrt(k_values * h) + normal_random_variables

# Convert x_k to a DataFrame for easier manipulation
df = pd.DataFrame(x_k, columns=['x_k'])


def calculate_trends(df: pd.DataFrame, window_sizes):
    trends = {}
    for window in window_sizes:
        trends[f'moving_average_{window}'] = df['x_k'].rolling(window=window, min_periods=1, center=True).mean()
        trends[f'moving_median_{window}'] = df['x_k'].rolling(window=window, min_periods=1, center=True).median()
    return trends


def turning_points(df):
    p = 0
    n = len(df)
    for i in range(0, n - 2):
        if (df[i] < df[i + 1]) and (df[i + 1] > df[i + 2]) \
            or (df[i] > df[i + 1]) and (df[i + 1] < df[i + 2]):
            p += 1
    return p, 2 * (n - 2) / 3, math.sqrt((16 * n - 29) / 90)


def kendall_rank(df):
    p = 0
    n = len(df)
    for i in range(0, n - 1):
        for j in range(i + 1, n):
            if df[i] < df[j]:
                p += 1
    return 4 * p / (n * (n - 1)) - 1, 0, math.sqrt(2 * (2 * n + 5) / (9 * n * (n - 1)))


window_sizes = [21, 51, 111]
trends = calculate_trends(df, window_sizes)
results = {}

for key, trend in trends.items():
    residuals = df['x_k'] - trend
    # Calculate Kendall's Tau coefficient (tau, p-value)
    p, Ep, Sp = turning_points(residuals)
    print(p, Ep, Sp)
    if (p > Ep - 3 * Sp) and (p <= Ep + 3 * Sp):
        print(f"Ряд остатков {key} по поворотным точкам случаен")
    else:
        print(f"Ряд остатков {key} по поворотным точкам не случаен")
    t, Et, St = kendall_rank(residuals)
    print(t, Et, St)
    if (t > Et - 3 * St) and (t <= Et + 3 * St):
        print(f"Ряд остатков {key} по коэффициенту Кендала случаен")
    else:
        print(f"Ряд остатков {key} по коэффициенту Кендала не случаен")

