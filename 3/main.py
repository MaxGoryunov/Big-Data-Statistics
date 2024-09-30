import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import scipy.stats as stats
from scipy.fft import fft
from common import turning_points_statistics, kendall_rank_statistics, chi_squared_statistics

# Constants
h = 0.1
k_values = np.arange(0, 501)
num_coefficients = [0.01, 0.05, 0.1, 0.3]

# np.random.seed(42)  # For reproducibility
norm_random = np.random.normal(0, 1, len(k_values))
x_k = 0.5 * np.sin(k_values * h) + norm_random


def exponential_moving_average(data, alpha):
    ema = np.zeros_like(data)
    for i in range(len(data)):
        if i == 0:
            ema[i] = data[i]
        else:
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
    return ema


trends = {}
for alpha in num_coefficients:
    trends[alpha] = exponential_moving_average(x_k, alpha)

# 3. Compare resulting trends with actual values
plt.figure(figsize=(12, 6))
plt.plot(k_values, x_k, label='Значения', alpha=0.5)
for alpha, trend in trends.items():
    plt.plot(k_values, trend, label=f'Тренд (alpha={alpha})')
plt.title('Сравнение трендов')
plt.xlabel('k')
plt.ylabel('Значение')
plt.legend()
plt.show()

# 4. Calculate amplitude spectrum and major frequency
spectrum = np.abs(fft(x_k))
frequencies = np.fft.fftfreq(len(x_k), d=h)

# Get the positive frequency part
positive_frequencies = frequencies[:len(frequencies)//2]
positive_spectrum = spectrum[:len(spectrum)//2]

# Finding the peak frequency
peak_indices, _ = find_peaks(positive_spectrum)
major_frequency = positive_frequencies[peak_indices[np.argmax(positive_spectrum[peak_indices])]]

# Plot amplitude spectrum
plt.figure(figsize=(12, 6))
plt.plot(positive_frequencies, positive_spectrum)
plt.title('Амплитудный спектр')
plt.xlabel('Частота')
plt.ylabel('Амплитуда')
plt.grid()
plt.show()

print(f'Главная частота: {major_frequency}')

# 5. Subtract trends from the series and analyze residuals
residuals = {alpha: x_k - trend for alpha, trend in trends.items()}

for alpha, res in residuals.items():
    plt.figure(figsize=(12, 6))
    plt.plot(k_values, res, label=f'Остатки (alpha={alpha})')
    plt.title(f'Остатки после вычета тренда (alpha={alpha})')
    plt.xlabel('k')
    plt.ylabel('Значение')
    plt.axhline(0, color='red', linestyle='--', label='Ноль')
    plt.legend()
    plt.show()
    mean_residual = np.mean(res)
    print(f'Среднее (alpha={alpha}): {mean_residual}')
    dev_residual = np.std(res, ddof=1)
    print(f'СКО (alpha={alpha}): {dev_residual}')
    turning_points_statistics(res, f"alpha={alpha}")
    kendall_rank_statistics(res, f"alpha={alpha}")
    chi_squared_statistics(res, mean_residual, dev_residual, f'alpha={alpha}')
    print("---------------")
