import numpy as np
import matplotlib.pyplot as plt

def line_distance(params, X):
    a, b, d = params
    return np.abs(a * X[:, 0] + b * X[:, 1] + d) / np.sqrt(a**2 + b**2)

def ransac_line(X, num_iterations=1000, threshold=0.6, min_inliers=25):
    best_inliers = np.empty((0, 2))
    best_params = None
    n = len(X)
    for _ in range(num_iterations):
        idx = np.random.choice(n, 2, replace=False)
        p1, p2 = X[idx]
        a, b = p2[1] - p1[1], p1[0] - p2[0]
        d = -(a * p1[0] + b * p1[1])
        norm = np.sqrt(a**2 + b**2)
        if norm == 0:
            continue
        a, b, d = a / norm, b / norm, d / norm
        dist = line_distance([a, b, d], X)
        inliers = X[dist < threshold]
        if len(inliers) > len(best_inliers) and len(inliers) >= min_inliers:
            best_inliers = inliers
            best_params = [a, b, d]
    return best_params, best_inliers

def circle_distance(params, X):
    x0, y0, r = params
    return np.abs(np.sqrt((X[:, 0] - x0)**2 + (X[:, 1] - y0)**2) - r)

def ransac_circle(X, num_iterations=1000, threshold=1.5, min_inliers=30):
    best_inliers = np.empty((0, 2))
    best_params = None
    n = len(X)
    for _ in range(num_iterations):
        idx = np.random.choice(n, 3, replace=False)
        p1, p2, p3 = X[idx]
        A = np.array([[p1[0], p1[1], 1],
                      [p2[0], p2[1], 1],
                      [p3[0], p3[1], 1]])
        B = np.array([[-(p1[0]**2 + p1[1]**2)],
                      [-(p2[0]**2 + p2[1]**2)],
                      [-(p3[0]**2 + p3[1]**2)]])
        if np.linalg.matrix_rank(A) < 3:
            continue
        C = np.linalg.solve(A, B)
        x0, y0 = -0.5 * C[0, 0], -0.5 * C[1, 0]
        r = np.sqrt((x0**2 + y0**2) - C[2, 0])
        dist = circle_distance([x0, y0, r], X)
        inliers = X[dist < threshold]
        if len(inliers) > len(best_inliers) and len(inliers) >= min_inliers:
            best_inliers = inliers
            best_params = [x0, y0, r]
    return best_params, best_inliers


np.random.seed(0)
N = 120
half = N // 2

# Circle points
r_true, x0_true, y0_true = 10, 2, 3
theta = np.linspace(0, 2 * np.pi, half)
x_circ = x0_true + (r_true + np.random.randn(half) * 0.5) * np.cos(theta)
y_circ = y0_true + (r_true + np.random.randn(half) * 0.5) * np.sin(theta)
X_circ = np.c_[x_circ, y_circ]

# Line points
m_true, c_true = -1, 2
x_line = np.linspace(-12, 12, half)
y_line = m_true * x_line + c_true + np.random.randn(half)
X_line = np.c_[x_line, y_line]

X = np.vstack((X_circ, X_line))
np.random.shuffle(X)

best_line_params, line_inliers = ransac_line(X)
if best_line_params is None:
    best_line_params, line_inliers = ransac_line(X, threshold=1.0, min_inliers=20)

remaining = np.array([p for p in X if not any(np.allclose(p, q) for q in line_inliers)])
best_circle_params, circle_inliers = ransac_circle(remaining)

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(X[:, 0], X[:, 1], s=10, color='gray', label='All Points')

ax.scatter(line_inliers[:, 0], line_inliers[:, 1], s=15, color='red', label='Line Inliers')
if circle_inliers.size > 0:
    ax.scatter(circle_inliers[:, 0], circle_inliers[:, 1], s=15, color='blue', label='Circle Inliers')

ax.add_patch(plt.Circle((x0_true, y0_true), r_true, color='limegreen', fill=False, linestyle='--', label='GT Circle'))
x_vals = np.linspace(-12, 12, 100)
ax.plot(x_vals, m_true * x_vals + c_true, 'g--', label='GT Line')

if best_line_params is not None:
    a, b, d = best_line_params
    ax.plot(x_vals, -(a * x_vals + d) / b, 'r-', label='Estimated Line')
if best_circle_params is not None:
    x0, y0, r = best_circle_params
    ax.add_patch(plt.Circle((x0, y0), r, color='b', fill=False, label='Estimated Circle'))

ax.set_aspect('equal', adjustable='box')
ax.legend()
ax.set_title("RANSAC Line and Circle Fitting")
plt.tight_layout()
plt.savefig('figures/q2_ransac_final.png', dpi=300)
plt.show()

if best_line_params is not None:
    print(f"Estimated Line: a={a:.3f}, b={b:.3f}, d={d:.3f}")
if best_circle_params is not None:
    print(f"Estimated Circle: x0={x0:.3f}, y0={y0:.3f}, r={r:.3f}")
