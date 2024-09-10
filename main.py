# This is a sample Python script.
import math
from functools import reduce
from math import sqrt


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def first_norm(vec: list):
    return reduce(lambda cum, cur: cum + abs(cur), vec, 0)


def second_norm(vec: list):
    return sqrt(reduce(lambda cum, cur: cum + cur ** 2, vec, 0))


def infinite_norm(vec: list):
    return max([abs(val) for val in vec])


def l1_weighted(vec: list, w: list):
    if sum([abs(val) for val in w]) == 1:
        print("Weights satisfy L1 requirements")
    else:
        print("Weights do not satisfy L1 requirements")
    return sum([abs(a * b) for a, b in zip(vec, w)])


def l2_weighted(vec: list, w: list):
    if sum([abs(val) ** 2 for val in w]) ** 0.5 == 1:
        print("Weights satisfy L2 requirements")
    else:
        print("Weights do not satisfy L2 requirements")
    return sum([abs(a * b) ** 2 for a, b in zip(vec, w)]) ** 0.5


def l_inf_weighted(vec: list, w: list):
    return max([abs(a * b) for a, b in zip(vec, w)])


def factorial_check():
    try:
        value = int(input("Enter an int to calculate factorial: "))
        if value < 0:
            print("Invalid input")
            return
        print(f"{value}! == ", math.factorial(value))
    except ValueError:
        print("This is not an int!")


def is_float_vec(vec: list):
    for val in vec:
        try:
            float(val)
        except ValueError:
            return False
    return True


def min_max_sum(vec: list):
    print(f"vector: {vec}")
    print(f"min: {min(vec)}")
    print(f"max: {max(vec)}")
    print(f"sum: {sum(vec)}")


def norm_test():
    vec = input("Enter your vector of length 5: ").split()
    if not is_float_vec(vec):
        print("Your vec values are not float")
        return
    vec = [float(val) for val in vec]
    if len(vec) != 5:
        print("Incorrect length")
    min_max_sum(vec)
    weights = input("Enter weights for vector of length 5: ").split()
    if not is_float_vec(weights):
        print("Your weights are not float")
        return
    weights = [float(val) for val in weights]
    if len(weights) != 5:
        print("Incorrect length")
    if any([i <= 0 for i in weights]):
        print("Coefficients must be positive")
    print(f"l_1 norm: {first_norm(vec)}")
    print(f"l_2 norm: {second_norm(vec)}")
    print(f"l_inf norm: {infinite_norm(vec)}")

    print(f"l_1 weighted: {l1_weighted(vec, weights)}")
    print(f"l_2 weighted: {l2_weighted(vec, weights)}")
    print(f"l_inf weighted: {l_inf_weighted(vec, weights)}")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    x = list(range(-10, 6))
    y = list(range(-5, 11))
    print(f"x: {x}, norm = {first_norm(x)}")
    print(f"y: {y}, norm = {first_norm(y)}")
    z = sorted([*x[::2], *y[1::2]])
    print(f"z: {z}, norm = {first_norm(z)}")
    factorial_check()
    norm_test()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
