import math
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


def turning_points(xs):
    p = 0
    n = len(xs)
    for i in range(0, n - 2):
        if (xs[i] < xs[i + 1]) and (xs[i + 1] > xs[i + 2]) \
            or (xs[i] > xs[i + 1]) and (xs[i + 1] < xs[i + 2]):
            p += 1
    return p, 2 * (n - 2) / 3, math.sqrt((16 * n - 29) / 90)


def turning_points_statistics(xs, label):
    p, Ep, Sp = turning_points(xs)
    print("Поворотные точки: ", p, Ep, Sp)
    if (p > Ep - 3 * Sp) and (p <= Ep + 3 * Sp):
        print(f"Ряд остатков {label} по поворотным точкам случаен")
    else:
        print(f"Ряд остатков {label} по поворотным точкам не случаен")


def kendall_rank(xs):
    p = 0
    n = len(xs)
    for i in range(0, n - 1):
        for j in range(i + 1, n):
            if xs[i] < xs[j]:
                p += 1
    return 4 * p / (n * (n - 1)) - 1, 0, math.sqrt(2 * (2 * n + 5) / (9 * n * (n - 1)))


def kendall_rank_statistics(xs, label):
    t, Et, St = kendall_rank(xs)
    print(t, Et, St)
    if (t > Et - 3 * St) and (t <= Et + 3 * St):
        print(f"Ряд остатков {label} по коэффициенту Кендала случаен")
    else:
        print(f"Ряд остатков {label} по коэффициенту Кендала не случаен")


def chi_squared_statistics(xs, mean, dev, label):
    n_samples = len(xs)
    # print(f"samples: {n_samples}")
    # bins = np.linspace(-4, 4, 21)  # 20 intervals from -4 to 4
    observed_freq, bins = np.histogram(xs)

    expected_freq = np.zeros_like(observed_freq)
    for i in range(len(bins) - 1):
        p = stats.norm.cdf(bins[i + 1], loc=mean, scale=dev) - stats.norm.cdf(bins[i], loc=mean, scale=dev)
        expected_freq[i] = p * n_samples
        # expected_freq[i] *= n_samples
        # print(i, diff)
        # print(expected_freq[i], diff * n_samples)
    # expected_freq *= n_samples  # Scale by the total number of samples
    # print(expected_freq)
    chi2_statistic = np.sum((observed_freq - expected_freq) ** 2 / expected_freq)
    degrees_of_freedom = len(bins) - 1 - 1
    crit_value = stats.chi2.ppf(0.95, df=degrees_of_freedom)
    print(f'Статистика Хи^2 для {label}: {chi2_statistic}')
    print(f'Степеней свободы: {degrees_of_freedom}')
    if (chi2_statistic < crit_value):
        print(f"Данные {label} нормально распределены")
    else:
        print(f"Данные {label} не распределены нормально")
