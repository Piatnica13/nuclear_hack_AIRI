import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# ============================================================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================================

CSV_PATH = "step_50_metrics (2).csv"   # <-- если нужно, поменяй имя файла

df = pd.read_csv(CSV_PATH)

df = df.dropna(subset=["global_step", "val_nrmse_quick"])

X = df["global_step"].to_numpy(dtype=float)
Y = df["val_nrmse_quick"].to_numpy(dtype=float)

print("=" * 60)
print(f"Загружено {len(X)} точек")
print(f"global_step: {X.min():.0f} ... {X.max():.0f}")
print(f"NRMSE: {Y.min():.6f} ... {Y.max():.6f}")
print("=" * 60)

# ============================================================
# 2. НАЧАЛЬНЫЕ ПРИБЛИЖЕНИЯ
# ============================================================

a0 = np.mean(Y[-5:]) * 0.98
b0 = max(Y[0] - a0, 1e-6)

x_mid = X[len(X)//2]
y_mid = Y[len(Y)//2]

ratio = (y_mid - a0) / b0
ratio = np.clip(ratio, 1e-6, 0.999)

c0 = -np.log(ratio) / np.log(x_mid)
c0 = np.clip(c0, 0.1, 3.0)

p0 = [a0, b0, c0]

print("\nНачальные параметры:")
print(f"a = {a0:.6f}")
print(f"b = {b0:.6f}")
print(f"c = {c0:.6f}")

# ============================================================
# 3. СТЕПЕННАЯ МОДЕЛЬ
# ============================================================

def power_law(x, a, b, c):
    return a + b * x ** (-c)

popt, pcov = curve_fit(
    power_law,
    X,
    Y,
    p0=p0,
    bounds=([0, 0, 0], [np.inf, np.inf, np.inf]),
    maxfev=20000
)

a, b, c = popt

perr = np.sqrt(np.diag(pcov))
a_err, b_err, c_err = perr

# ============================================================
# 4. КАЧЕСТВО АППРОКСИМАЦИИ
# ============================================================

Y_fit = power_law(X, a, b, c)

ss_res = np.sum((Y - Y_fit) ** 2)
ss_tot = np.sum((Y - np.mean(Y)) ** 2)

r2 = 1 - ss_res / ss_tot

# ============================================================
# 5. КРИТЕРИИ ПЛАТО
# ============================================================

def find_plateau_fraction(epsilon):
    return (b / (epsilon * (Y[0] - a))) ** (1 / c)


def find_plateau_derivative(threshold):
    return (b * c * 100 / threshold) ** (1 / (c + 1))


x_95 = find_plateau_fraction(0.05)
x_99 = find_plateau_fraction(0.01)

rel_threshold = 0.005

y_ref = power_law(X[-1], a, b, c)
abs_threshold = rel_threshold * y_ref

x_der_rel = find_plateau_derivative(abs_threshold)

growth_abs = 0.0001
x_der_abs = find_plateau_derivative(growth_abs)

X_opt = int(np.ceil(max(
    x_95,
    x_99,
    x_der_rel,
    x_der_abs
)))

# ============================================================
# 6. ГРАФИК
# ============================================================

X_smooth = np.linspace(
    X.min(),
    max(X_opt * 1.2, X.max() * 2),
    1000
)

Y_smooth = power_law(X_smooth, a, b, c)

plt.figure(figsize=(14,7))

plt.scatter(
    X,
    Y,
    s=30,
    color="red",
    label="Измерения",
    zorder=10
)

plt.plot(
    X_smooth,
    Y_smooth,
    linewidth=2,
    color="blue",
    label="Степенная модель"
)

plt.axhline(
    a,
    linestyle="--",
    color="gray",
    label=f"Асимптота = {a:.6f}"
)

plt.axvline(
    x_95,
    color="orange",
    linestyle="--",
    label=f"95%: {x_95:.0f}"
)

plt.axvline(
    x_99,
    color="darkorange",
    linestyle="--",
    label=f"99%: {x_99:.0f}"
)

plt.axvline(
    x_der_rel,
    color="purple",
    linestyle="--",
    label=f"<0.5%/100"
)

plt.axvline(
    x_der_abs,
    color="magenta",
    linestyle="--",
    label=f"<0.0001/100"
)

plt.axvline(
    X_opt,
    linewidth=3,
    color="green",
    alpha=0.4,
    label=f"Рекомендуемое плато = {X_opt}"
)

plt.scatter(
    [X_opt],
    [power_law(X_opt, a, b, c)],
    s=180,
    marker="D",
    color="green",
    edgecolors="black",
    zorder=20
)

plt.grid(alpha=0.3)

plt.xlabel("Global step")
plt.ylabel("Validation NRMSE")
plt.title("Learning Curve Plateau Analysis")

plt.legend()

plt.tight_layout()

plt.savefig(
    "learning_curve_plateau.png",
    dpi=300
)

plt.show()

# ============================================================
# 7. ИТОГ
# ============================================================

improvement = b * c * X_opt ** (-(c + 1)) * 100

print("\n")
print("=" * 70)
print("РЕЗУЛЬТАТЫ")
print("=" * 70)

print(f"Модель:")
print(f"NRMSE = {a:.6f} + {b:.6f} * X^(-{c:.4f})")

print()

print(f"R² = {r2:.6f}")

print()

print(f"Асимптота:")
print(f"{a:.6f} ± {a_err:.6f}")

print()

print("Критерии плато:")

print(f"95% пути к пределу:              {x_95:.0f}")
print(f"99% пути к пределу:              {x_99:.0f}")
print(f"Прирост <0.5% за 100 шагов:      {x_der_rel:.0f}")
print(f"Прирост <0.0001 за 100 шагов:    {x_der_abs:.0f}")

print()

print("=" * 70)
print(f"РЕКОМЕНДУЕМЫЙ МИНИМАЛЬНЫЙ РАЗМЕР ДАТАСЕТА = {X_opt}")
print("=" * 70)

print()

print(f"Ожидаемый NRMSE: {power_law(X_opt, a, b, c):.6f}")
print(f"Прирост за 100 шагов: {improvement:.8f}")

print("\nГрафик сохранен как learning_curve_plateau.png")