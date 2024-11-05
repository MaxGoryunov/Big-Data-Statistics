import numpy as np
import statsmodels.api as sm


n = 20
x1 = np.random.normal(0, 1, n)
x2 = np.random.normal(0, 1, n)
x3 = np.random.normal(0, 1, n)
e = np.random.normal(0, 1, n)
y = 1 + 3 * x1 - 2 * x2 + x3 + e
print("x1", x1)
print("x2", x2)
print("x3", x3)
print("y", y)


X = np.column_stack((x1, x2, x3))
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()




rss = np.sum(model.resid ** 2)
rse = np.sqrt(rss / (len(y) - len(model.params)))
r_squared = model.rsquared


print(f"RSS: {rss:.4f}")
print(f"RSE: {rse:.4f}")
print(f"R^2: {r_squared:.4f}")


if r_squared > 0.8:
    conclusion = "Линейная модель объясняет большую долю дисперсии отклика."
elif r_squared > 0.5:
    conclusion = "Линейная модель объясняет среднюю долю дисперсии отклика."
else:
    conclusion = "Линейная модель объясняет малую долю дисперсии отклика."

print(conclusion)
