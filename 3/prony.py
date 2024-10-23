import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from numpy import log
from numpy.fft import fft, fftfreq


def hurst_exponent(time_series):
    n = len(time_series)
    logy = []
    logz = []
    for i in range(2, n + 1):
        sub_series = time_series[:i]
        mean = np.mean(sub_series)
        X = np.cumsum(sub_series - mean)
        R = np.max(X) - np.min(X)
        S = np.std(sub_series)
        # print(X, R, S)
        R_S = R / S
        logy.append(np.log(R_S))
        logz.append(np.log(i))
    # print(logz, logy)
    coeffs = np.polyfit(logz, logy, 1)
    nplogz = np.array(logz)

    # plt.figure(figsize=(12, 6))
    # plt.plot(logz, logy, linestyle='-', color='grey')
    # plt.plot(nplogz, nplogz * coeffs[0] + coeffs[1], linestyle='--', color='red')
    # plt.title(f'H = {coeffs[0]}')
    # plt.grid()
    # plt.show()
    return coeffs[0]
    # mean = np.mean(time_series)
    # X = np.cumsum(time_series - mean)
    # R = np.max(X) - np.min(X)
    # S = np.std(time_series)
    # R_S = R / S
    # print(R_S)
    # return log(R_S) / log(n)


df = pd.read_csv('temp.csv', parse_dates=['Дата'])
df = df.sort_values(by='Дата')


n_intervals = 10
time_series = df['Средняя температура'].values
intervals = np.array_split(time_series, n_intervals)
hurst_values = [hurst_exponent(interval) for interval in intervals]
alpha = 0.07
df['trend'] = df['Средняя температура'].ewm(alpha=alpha).mean()


fig_time_series = go.Figure()


fig_time_series.add_trace(go.Scatter(
    x=df['Дата'],
    y=df['Средняя температура'],
    mode='lines',
    name='Временной ряд'
))

for i, interval in enumerate(intervals):
    interval_start = df['Дата'].iloc[len(interval) * i]
    interval_end = df['Дата'].iloc[len(interval) * (i + 1) - 1]

    fig_time_series.add_shape(type="line",
                              x0=interval_start, x1=interval_start, y0=min(df['Средняя температура']),
                              y1=max(df['Средняя температура']),
                              line=dict(color="LightSeaGreen", width=2, dash="dash")
                              )

    fig_time_series.add_shape(type="line",
                              x0=interval_end, x1=interval_end, y0=min(df['Средняя температура']),
                              y1=max(df['Средняя температура']),
                              line=dict(color="LightSeaGreen", width=2, dash="dash")
                              )

    interval_middle = interval_start + (interval_end - interval_start) / 2
    fig_time_series.add_trace(go.Scatter(
        x=[interval_middle],
        y=[np.mean(interval)],
        mode='text',
        text=[f'H={hurst_values[i]:.2f}'],
        showlegend=False
    ))

fig_time_series.add_trace(go.Scatter(
    x=df['Дата'],
    y=df['trend'],
    mode='lines',
    name='Тренд',
    line=dict(color='red', width=2)
))

fig_time_series.update_layout(
    title='Временной ряд температур',
    xaxis_title='Дата',
    yaxis_title='Средняя температура'
)

fig_time_series.show()


def search_coeff_a(x_k: list, m: int):
    X_matrix = []

    n = len(x_k)
    i = m
    while i != n:
        row = []
        for k in range(m, 0, -1):
            row.append(x_k[i - k])
        X_matrix.append(row)

        i += 1

    X_matrix = np.array(X_matrix)

    b_vect = np.array([-x_k[i] for i in range(m, n)])

    a_coef = np.linalg.lstsq(X_matrix, b_vect, rcond=None)[0]

    return a_coef


def search_h_k(x_list, z_list, m):
    Z_matrix = []
    for k in range(len(x_list)):
        row = []
        for i in range(m):
            row.append(pow(z_list[i], k - 1))
        Z_matrix.append(row)

    Z_matrix = np.array(Z_matrix)
    b_vect = np.array(x_list)

    h_k = np.linalg.lstsq(Z_matrix, b_vect, rcond=None)[0]

    return h_k


def method_proni(x_list, m):
    # 1-ый этап процедуры
    a_coef = search_coeff_a(x_list, m)

    # 2-ой этап процедуры
    coef_ = np.insert(a_coef, 0, 1)
    z_k = np.roots(coef_)
    # print(z_k)

    lambda_k_t = [np.log(np.abs(z)) for z in z_k]  # коэффициент затухания
    omega_k_t = [math.atan(z.imag / z.real) / (2 * np.pi) for z in z_k]  # частоты

    # 3-ий этап процедуры
    h_k = search_h_k(x_list, z_k, m)

    A_k = [np.abs(h) for h in h_k]  # амплитуды
    phi_k = [math.atan(h.imag / h.real) for h in h_k]  # фазы

    x_proni = [
        sum([A_k[i] * np.exp(complex(-lambda_k_t[i] * k, omega_k_t[i] * k + phi_k[i])) for i in range(m)])
        for k in range(len(x_list))
    ]
    # print(x_proni)
    return x_proni
    # return A_k, omega_k_t


y_proni = method_proni(df['Средняя температура'].tolist(), 3)
x_list = [k for k in range(len(y_proni))]
k_list = [k / (len(x_list) // 2) for k in x_list]
x_pr = [el.real for el in y_proni]
y_pr = [el.imag for el in y_proni]
    # plt.scatter(x_pr, y_pr, color='black', marker='.')
plt.plot(k_list, x_pr)
plt.title("Метод Прони")
plt.xlabel('Частота 1/год')
plt.show()
