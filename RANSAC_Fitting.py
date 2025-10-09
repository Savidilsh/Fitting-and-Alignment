import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

# --- Data Generation ---
np.random.seed(0)
N, r_gt, x0_gt, y0_gt = 100, 10, 2, 3
half_n = N // 2
t = np.random.uniform(0, 2 * np.pi, half_n)
n = (r_gt / 16) * np.random.randn(half_n)
x_circ, y_circ = x0_gt + (r_gt + n) * np.cos(t), y0_gt + (r_gt + n) * np.sin(t)
X_circ = np.hstack((x_circ.reshape(half_n, 1), y_circ.reshape(half_n, 1)))
m_gt, b_gt = -1, 2
x_line_pts = np.linspace(-12, 12, half_n)
y_line_pts = m_gt * x_line_pts + b_gt + np.random.randn(half_n)
X_line = np.hstack((x_line_pts.reshape(half_n, 1), y_line_pts.reshape(half_n, 1)))
X = np.vstack((X_circ, X_line))

# --- RANSAC Implementations ---
def fit_line_ransac(data, threshold=0.5, n_iterations=1000):
    best_inliers_idx = []
    for _ in range(n_iterations):
        p1, p2 = data[np.random.choice(data.shape[0], 2, replace=False)]
        a, b = p2[1] - p1[1], p1[0] - p2[0]
        norm = np.sqrt(a**2 + b**2)
        a, b = a / norm, b / norm
        d = -(a * p1[0] + b * p1[1])
        distances = np.abs(a * data[:, 0] + b * data[:, 1] + d)
        inliers_idx = np.where(distances < threshold)[0]
        if len(inliers_idx) > len(best_inliers_idx):
            best_inliers_idx = inliers_idx
    # Refit model to all inliers
    inlier_points = data[best_inliers_idx]
    centroid = np.mean(inlier_points, axis=0)
    _, _, Vt = linalg.svd(inlier_points - centroid)
    a, b = Vt[1]
    d = -(a * centroid[0] + b * centroid[1])
    return (a, b, d), best_inliers_idx

def fit_circle_ransac(data, threshold=0.8, n_iterations=1000):
    best_inliers_idx, best_model = [], None
    for _ in range(n_iterations):
        p1, p2, p3 = data[np.random.choice(data.shape[0], 3, replace=False)]
        A = np.array([[2*(p2[0]-p1[0]), 2*(p2[1]-p1[1])], [2*(p3[0]-p2[0]), 2*(p3[1]-p2[1])]])
        B = np.array([p2[0]**2+p2[1]**2-p1[0]**2-p1[1]**2, p3[0]**2+p3[1]**2-p2[0]**2-p2[1]**2])
        try: xc, yc = np.linalg.solve(A, B)
        except np.linalg.LinAlgError: continue
        r = np.sqrt((p1[0]-xc)**2 + (p1[1]-yc)**2)
        distances = np.abs(np.sqrt((data[:, 0]-xc)**2 + (data[:, 1]-yc)**2) - r)
        inliers_idx = np.where(distances < threshold)[0]
        if len(inliers_idx) > len(best_inliers_idx):
            best_inliers_idx, best_model = inliers_idx, (xc, yc, r)
    return best_model, best_inliers_idx

# --- Execution & Visualization ---
line_model, line_inliers_idx = fit_line_ransac(X)
X_remnant = np.delete(X, line_inliers_idx, axis=0)
circle_model, circle_inliers_idx_local = fit_circle_ransac(X_remnant)
# Find original indices for circle inliers
all_indices = np.arange(X.shape[0])
remnant_indices = np.delete(all_indices, line_inliers_idx)
circle_inliers_idx = remnant_indices[circle_inliers_idx_local]

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.plot(X[:, 0], X[:, 1], '.', color='#17becf', label='All points')
ax.plot(X[line_inliers_idx, 0], X[line_inliers_idx, 1], 'o', color='#98df8a', label='Line Inliers')
ax.plot(X[circle_inliers_idx, 0], X[circle_inliers_idx, 1], 'o', color='#1f77b4', label='Circle Inliers')
a, b, d = line_model
x_vals = np.array(ax.get_xlim())
y_vals = (-a * x_vals - d) / b
ax.plot(x_vals, y_vals, '-', color='#9467bd', label='RANSAC Line', linewidth=2)
xc, yc, r = circle_model
ax.add_patch(plt.Circle((xc, yc), r, color='#d62728', fill=False, label='RANSAC Circle', linewidth=2))
ax.set_aspect('equal', adjustable='box')
ax.legend()
plt.savefig('ransac_fit.png', bbox_inches='tight')
plt.show()