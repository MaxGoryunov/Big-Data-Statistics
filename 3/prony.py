import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from numpy import log
from numpy.fft import fft, fftfreq
import matplotlib.pyplot as plt


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
    plt.figure(figsize=(12, 6))
    plt.plot(logz, logy, linestyle='-', color='grey')
    nplogz = np.array(logz)
    plt.plot(nplogz, nplogz * coeffs[0] + coeffs[1], linestyle='--', color='red')
    plt.title(f'H = {coeffs[0]}')
    plt.grid()
    plt.show()
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
alpha = 0.05
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

from numpy.fft import fft, fftfreq
trend = df['trend'].values
n = len(trend)
t = (df['Дата'].max() - df['Дата'].min()).days / n

frequencies = fftfreq(n, t)
fft_values = fft(trend)

positive_freqs = frequencies[:n // 2]
positive_fft_values = np.abs(fft_values[:n // 2]) / n

fig_spectrum = go.Figure()

fig_spectrum.add_trace(go.Bar(
    x=positive_freqs,
    y=positive_fft_values,
    name="Спектр Прони"
))

fig_spectrum.update_layout(
    title="Спектр Прони для тренда",
    xaxis_title="Частоты",
    yaxis_title="Амплитуда"
)

fig_spectrum.show()
