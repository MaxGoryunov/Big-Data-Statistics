import numpy as np
import pandas as pd

# Parameters
M = 10000  # Number of Monte Carlo tests
n = 100  # Sample size


def huber(x, k=1.44):
    return np.mean(np.where(np.abs(x) <= k, x, k * np.sign(x)))


# Functions to generate samples and calculate means and medians
def simulate_samples():
    # Store results for each distribution
    means = {'Нормальное': [], 'Коши': [], 'Смесь': []}
    medians = {'Нормальное': [], 'Коши': [], 'Смесь': []}
    huber_means = {'Нормальное': [], 'Коши': [], 'Смесь': []}
    cleaned_means = {'Нормальное': [], 'Коши': [], 'Смесь': []}

    for _ in range(M):
        # Sample from Normal distribution
        normal_sample = np.random.normal(0, 1, n)
        means['Нормальное'].append(np.mean(normal_sample))
        medians['Нормальное'].append(np.median(normal_sample))
        huber_means['Нормальное'].append(huber(normal_sample))
        cleaned_means['Нормальное'].append(remove_outliers_and_get_mean(normal_sample))

        # Sample from Cauchy distribution
        cauchy_sample = np.random.standard_cauchy(size=n)
        means['Коши'].append(np.mean(cauchy_sample))
        medians['Коши'].append(np.median(cauchy_sample))
        huber_means['Коши'].append(huber(cauchy_sample))
        cleaned_means['Коши'].append(remove_outliers_and_get_mean(cauchy_sample))

        # Sample from the mixture distribution
        mixture_sample = 0.9 * np.random.normal(0, 1, n) + 0.1 * np.random.standard_cauchy(size=n)
        means['Смесь'].append(np.mean(mixture_sample))
        medians['Смесь'].append(np.median(mixture_sample))
        huber_means['Смесь'].append(huber(mixture_sample))
        cleaned_means['Смесь'].append(remove_outliers_and_get_mean(mixture_sample))

    return means, medians, cleaned_means, huber_means


def remove_outliers_and_get_mean(data):
    """Remove outliers using boxplot method and calculate mean of remaining values."""
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    cleaned_data = data[(data >= lower_bound) & (data <= upper_bound)]

    return np.mean(cleaned_data) if len(cleaned_data) > 0 else np.nan


# Calculate mean and variance
def calculate_statistics(means, medians, cleaned_means, huber_means):
    results_means = {}
    results_medians = {}
    results_cleaned_means = {}
    results_huber_means = {}

    for dist in means.keys():
        results_means[dist] = {
            'Mean': np.mean(means[dist]),
            'Variance': np.var(means[dist])
        }
        results_medians[dist] = {
            'Mean': np.mean(medians[dist]),
            'Variance': np.var(medians[dist])
        }
        results_cleaned_means[dist] = {
            'Mean': np.mean(cleaned_means[dist]),
            'Variance': np.var(cleaned_means[dist])
        }
        results_huber_means[dist] = {
            'Mean': np.mean(huber_means[dist]),
            'Variance': np.var(huber_means[dist])
        }

    return results_means, results_medians, results_cleaned_means, results_huber_means


# Main simulation
if __name__ == "__main__":
    means, medians, cleaned_means, huber_means = simulate_samples()
    results_means, results_medians, results_cleaned_means, results_huber_means = calculate_statistics(
        means, medians, cleaned_means, huber_means)

    # Print results
    print("Среднее и дисперсия выборочных средних:")
    for dist, stats in results_means.items():
        print(f"{dist}: Среднее = {stats['Mean']:.4f}, Дисперсия = {stats['Variance']:.4f}")

    print("\nСреднее и дисперсия выборочных медиан:")
    for dist, stats in results_medians.items():
        print(f"{dist}: Среднее = {stats['Mean']:.4f}, Дисперсия = {stats['Variance']:.4f}")

    print("\nСреднее и дисперсия оценки Хубера:")
    for dist, stats in results_huber_means.items():
        print(f"{dist}: Среднее = {stats['Mean']:.4f}, Дисперсия = {stats['Variance']:.4f}")

    print("\nСреднее и дисперсия двухэтапной оценки:")
    for dist, stats in results_cleaned_means.items():
        print(f"{dist}: Среднее = {stats['Mean']:.4f}, Дисперсия = {stats['Variance']:.4f}")
