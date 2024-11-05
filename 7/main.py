import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.pipeline import make_pipeline


data_points = np.array([[-2, -7], [-1, 0], [0, 1], [1, 2], [2, 9]])
x = data_points[:, 0].reshape(-1, 1)
y = data_points[:, 1]
print("x", x)
print("y", y)


def plot_regression(x, y, model_name: str, lambdas, noise_std=None, degree=11, true_degree=3):
    plt.figure(figsize=(12, 6))
    x_range = np.linspace(-2.1, 2.1, 100).reshape(-1, 1)
    poly = PolynomialFeatures(degree=true_degree)
    x_poly = poly.fit_transform(x)
    poly_model = LinearRegression().fit(x_poly, y)
    title = f'{model_name} Полиномиальная регрессия (p={degree})' + (f' с шумом СКО = {noise_std}' if noise_std else '')
    print(poly_model.coef_)
    x_poly_range = PolynomialFeatures(true_degree).fit_transform(x_range)
    y_poly_range = poly_model.predict(x_poly_range)
    plt.plot(x_range, y_poly_range, label="Модельный полином")
    # plt.show()
    for lambd in lambdas:
        if model_name == 'Ridge':

            model = make_pipeline(PolynomialFeatures(degree=degree), Ridge(alpha=lambd))
        elif model_name == 'Lasso':
            model = make_pipeline(PolynomialFeatures(degree=degree), Lasso(alpha=lambd))

        model.fit(x, y)
        y_pred = model.predict(x_range)
        coeffs = np.array(model.named_steps[model_name.lower()].coef_)
        coeffs[0] = model.named_steps[model_name.lower()].intercept_
        new_title = title + f"lambda={lambd}"
        print(new_title)
        for i, coeff in enumerate(coeffs):
            print(i, coeff)

        plt.plot(x_range, y_pred, label=f'{model_name} (lambda={lambd})')
    plt.scatter(x, y, color='red', label='Данные')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid()
    plt.show()


lambdas = [0.01, 0.1, 1, 10]

plot_regression(x, y, 'Ridge', lambdas)
plot_regression(x, y, 'Lasso', lambdas)


noise_stds = [0.1, 0.2, 0.3]

for noise_std in noise_stds:
    noise = np.random.normal(0, noise_std, y.shape)
    y_noisy = y + noise
    plot_regression(x, y_noisy, 'Ridge', lambdas, noise_std)
    plot_regression(x, y_noisy, 'Lasso', lambdas, noise_std)
