import numpy as np
from scipy.optimize import newton, minimize

def f(x): return (x - 3)**2 + 1
def df(x): return 2 * (x - 3)
def d2f(x): return 2.0

def gradient_descent(derivative, x0, lr=0.1, tol=1e-8):
    x = x0
    while True:
        x_new = x - lr * derivative(x)
        if abs(x_new - x) < tol:
            return x_new
        x = x_new

x0 = 0

print("2A")
print("Gradient descent:", gradient_descent(df, x0))
print("Newton:", newton(df, x0, fprime=d2f))
print("SLSQP:", minimize(f, x0, method="SLSQP").x[0])

def g(x): return x**4 - 3*x**2 + x + 5
def dg(x): return 4*x**3 - 6*x + 1
def d2g(x): return 12*x**2 - 6

for x0 in [0, 2]:
    gd = gradient_descent(dg, x0)
    nt = newton(dg, x0, fprime=d2g)
    sq = minimize(g, x0, method="SLSQP").x[0]

    print(f"\n2B, x0 = {x0}")
    print("Gradient descent:", gd)
    print("Newton:", nt)
    print("Newton g''(x):", d2g(nt))
    print("Newton:", "minimum" if d2g(nt) > 0 else "maximum")
    print("SLSQP:", sq)