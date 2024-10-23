import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



x_k = np.random.normal(0, 1, 195)
additional_values = [5, -4, 3.3, 2.99, -3]
x_k = np.concatenate((x_k, additional_values))
x_k = np.sort(x_k)

mean = np.mean(x_k)
std_dev = np.std(x_k)
lower_bound = mean - 3 * std_dev
upper_bound = mean + 3 * std_dev

outliers_sigma_rule = x_k[(x_k < lower_bound) | (x_k > upper_bound)]

plt.figure(figsize=(12, 6))
sns.boxplot(x=x_k)
plt.title('Боксплот')
plt.xlabel('Значения')
plt.grid()
plt.show()

Q1 = np.percentile(x_k, 25)
Q3 = np.percentile(x_k, 75)
IQR = Q3 - Q1
boxplot_lower_bound = Q1 - 1.5 * IQR
boxplot_upper_bound = Q3 + 1.5 * IQR
outliers_boxplot = x_k[(x_k < boxplot_lower_bound) | (x_k > boxplot_upper_bound)]

print("Среднее:", mean)
print("СКО:", std_dev)
print("Нижняя граница (3 сигмы):", lower_bound)
print("Верхняя граница (3 сигмы):", upper_bound)
print("Выбросы по трем сигмам:", outliers_sigma_rule)
print("\nНижняя граница (боксплот):", boxplot_lower_bound)
print("Верхняя граница (боксплот):", boxplot_upper_bound)
print("Выбросы по боксплоту:", outliers_boxplot)

# Comparing the two outlier results
common_outliers = set(outliers_sigma_rule).intersection(set(outliers_boxplot))
print("\nОбшие выбросы:", common_outliers)