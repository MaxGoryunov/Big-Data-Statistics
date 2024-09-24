import math
import statistics

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

# Step 1: Generate the data
# np.random.seed(42)  # For reproducibility
h = 0.05
k = np.arange(501)
noise = np.random.normal(0, 1, 501)
x_k = np.sqrt(k * h) + noise
truth = np.sqrt(k * h)

# Convert x_k to a DataFrame for easier manipulation
df = pd.DataFrame(x_k, columns=['x_k'])


def calculate_trends(df, sizes):
    trends = {}
    for window in sizes:
        temp = df['x_k']
        val = df['x_k'].rolling(window=window, min_periods=1, center=True).mean()
        val.iloc[0] = statistics.median([temp[0], temp[1], 3 * temp[1] - 2 * temp[2]])
        # print(len(val), len(temp), val.iloc[-1], temp.iloc[-1])
        val.iloc[-1] = statistics.median([temp.iloc[-1], temp.iloc[-2], 3 * temp.iloc[-2] - 2 * temp.iloc[-3]])
        trends[f'moving_average_{window}'] = val
        val = df['x_k'].rolling(window=window, min_periods=1, center=True).median()
        val.iloc[0] = statistics.median([temp[0], temp[1], 3 * temp[1] - 2 * temp[2]])
        val.iloc[-1] = statistics.median([temp.iloc[-1], temp.iloc[-2], 3 * temp.iloc[-2] - 2 * temp.iloc[-3]])
        trends[f'moving_median_{window}'] = val
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


# Визуализация результатов
plt.figure(figsize=(14, 8))
plt.plot(x_k, label="Исходный ряд", color='gray', alpha=0.6, linestyle='dashed')
plt.plot(truth, label="Истинный корень", color='red', alpha=0.6, linestyle='dashed')


for key, trend in trends.items():
    plt.plot(trend, label=f"{key}")



plt.title("Тренды по скользящим средним и медианам")
plt.legend()
plt.grid()
plt.show()

