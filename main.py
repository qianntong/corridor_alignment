"""
Full FNO Training + Auto Lambda Selection + Train/Test CSV Output
Corridor Alignment: USGS → Track Chart
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Paths
# ============================================================
data_path = "/Users/qianqiantong/PycharmProjects/corridor_alignment/corridor_data/"
output_path = "/Users/qianqiantong/PycharmProjects/corridor_alignment/output/"
os.makedirs(output_path, exist_ok=True)


# ============================================================
# 1. Load & Interpolate
# ============================================================
def load_and_interpolate(excel_usgs, excel_track, N=4096):
    df1 = pd.read_excel(excel_usgs)
    df2 = pd.read_excel(excel_track)

    x1, e1 = df1["total_dist_meters"].values, df1["elevation_meters"].values
    x2, e2 = df2["total_dist_meters"].values, df2["elevation_meters"].values

    L = min(x1.max(), x2.max())
    x_uniform = np.linspace(0, L, N)

    f1 = interp1d(x1, e1, kind="linear", fill_value="extrapolate")
    f2 = interp1d(x2, e2, kind="linear", fill_value="extrapolate")

    return x_uniform, f1(x_uniform), f2(x_uniform)


# ============================================================
# 2. Normalization
# ============================================================
def normalize_pair(e_usgs_train, e_true_train, e_usgs_test, e_true_test):
    mean = np.mean(np.concatenate([e_usgs_train, e_true_train]))
    std = np.std(np.concatenate([e_usgs_train, e_true_train])) + 1e-6

    return (
        (e_usgs_train - mean) / std,
        (e_true_train - mean) / std,
        (e_usgs_test - mean) / std,
        (e_true_test - mean) / std,
        mean,
        std
    )


# ============================================================
# 3. Keypoints
# ============================================================
def extract_keypoints(elevation, thr=0.01):
    peaks, _ = find_peaks(elevation, prominence=thr)
    valleys, _ = find_peaks(-elevation, prominence=thr)
    return np.sort(np.concatenate([peaks, valleys]))


# ============================================================
# 4. FNO1d Definition
# ============================================================
class SpectralConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, modes):
        super().__init__()
        self.modes = modes
        self.scale = 1/(in_ch*out_ch)
        self.weights = nn.Parameter(
            self.scale * torch.rand(in_ch, out_ch, modes, dtype=torch.cfloat)
        )

    def forward(self, x):
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(x.size(0), self.weights.size(1), x.size(-1)//2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes] = torch.einsum("bix,iox->box",
                                                 x_ft[:, :, :self.modes], self.weights)
        return torch.fft.irfft(out_ft, n=x.size(-1))


class FNO1d(nn.Module):
    def __init__(self, modes=32, width=64, layers=4):
        super().__init__()
        self.fc0 = nn.Linear(2, width)

        self.fourier_layers = nn.ModuleList([
            SpectralConv1d(width, width, modes) for _ in range(layers)
        ])
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(width, width, 1) for _ in range(layers)
        ])

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc0(x)
        x = x.permute(0,2,1)

        for f, c in zip(self.fourier_layers[:-1], self.conv_layers[:-1]):
            x = F.gelu(f(x) + c(x))

        x = self.fourier_layers[-1](x) + self.conv_layers[-1](x)
        x = x.permute(0,2,1)

        x = F.gelu(self.fc1(x))
        return self.fc2(x).squeeze(-1)


# ============================================================
# 5. Loss Functions
# ============================================================
def slope_loss(pred, true, dx):
    return torch.mean((torch.diff(pred)/dx - torch.diff(true)/dx)**2)

def keypoint_loss(pred, true, kp_idx):
    return torch.mean((pred[kp_idx] - true[kp_idx])**2)

def frequency_loss(pred, true, k=32):
    pf, tf = torch.fft.rfft(pred), torch.fft.rfft(true)
    return torch.mean(torch.abs(pf[:k] - tf[:k])**2)

def smoothness_loss(pred):
    d2 = torch.diff(pred, n=2)
    return torch.mean(d2**2)


# polynomial trend loss
def trend_loss(pred, true):
    x = torch.linspace(0, 1, pred.shape[-1], device=pred.device)
    X = torch.stack([x**3, x**2, x, torch.ones_like(x)], dim=1)

    theta_pred = torch.linalg.lstsq(X, pred.unsqueeze(1)).solution
    theta_true = torch.linalg.lstsq(X, true.unsqueeze(1)).solution
    return torch.mean((theta_pred - theta_true)**2)


# ============================================================
# 6. Lambda Auto Tune
# ============================================================
def sample_lambdas():
    return {
        "λ_slope": 10**random.uniform(-2, -1),
        "λ_kp":    10**random.uniform(-3.5, -1.5),
        "λ_freq":  10**random.uniform(-2.5, -1),
        "λ_trend": 10**random.uniform(-0.5, 0.5),
        "λ_smooth":10**random.uniform(-3, -1),
    }

def short_train_eval(model, X_train, Y_train, kp_idx, dx, lam, steps=80):
    """Train few steps to evaluate lambda combination"""
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    for _ in range(steps):
        opt.zero_grad()
        pred = model(X_train)

        L = torch.mean((pred - Y_train)**2)
        L += lam["λ_slope"]  * slope_loss(pred, Y_train, dx)
        L += lam["λ_kp"]     * keypoint_loss(pred[0], Y_train[0], kp_idx)
        L += lam["λ_freq"]   * frequency_loss(pred[0], Y_train[0])
        L += lam["λ_trend"]  * trend_loss(pred[0], Y_train[0])
        L += lam["λ_smooth"] * smoothness_loss(pred[0])

        L.backward()
        opt.step()

    # return validation proxy (trend is crucial)
    with torch.no_grad():
        final_pred = model(X_train)
        return float(torch.mean((final_pred - Y_train)**2)
                     + trend_loss(final_pred[0], Y_train[0]))


def find_best_lambdas(X_train, Y_train, kp_idx, dx, trials=12):
    results = []
    for i in range(trials):
        lam = sample_lambdas()
        model = FNO1d().to(device)
        score = short_train_eval(model, X_train, Y_train, kp_idx, dx, lam)
        results.append((score, lam))
        print(f"Trial {i+1}/{trials}: score={score:.4e}")
    results.sort(key=lambda x: x[0])
    return results[0][1]


# ============================================================
# 7. Load Training + Testing Data
# ============================================================
x_train, usgs_train, true_train = load_and_interpolate(
    data_path + "usgs_elevation_Clovis-Flagstaff_grade_data.xlsx",
    data_path + "tt_elevation_Clovis-Flagstaff_grade_data.xlsx",
)

x_test, usgs_test, true_test = load_and_interpolate(
    data_path + "usgs_elevation_Amarillo-FortWorth_grade_data.xlsx",
    data_path + "tt_elevation_Amarillo-FortWorth_grade_data.xlsx",
)

usgs_train_n, true_train_n, usgs_test_n, true_test_n, mean_e, std_e = normalize_pair(
    usgs_train, true_train, usgs_test, true_test
)

kp_idx = extract_keypoints(true_train_n)
dx = x_train[1] - x_train[0]

X_train = torch.tensor(np.stack([usgs_train_n, x_train/x_train.max()], axis=-1),
                       dtype=torch.float32).unsqueeze(0).to(device)
Y_train = torch.tensor(true_train_n, dtype=torch.float32).unsqueeze(0).to(device)

X_test = torch.tensor(np.stack([usgs_test_n, x_test/x_test.max()], axis=-1),
                      dtype=torch.float32).unsqueeze(0).to(device)
Y_test = torch.tensor(true_test_n, dtype=torch.float32).unsqueeze(0).to(device)


# ============================================================
# 8. Auto Lambda Search
# ============================================================
print("\n=== Auto-Tuning Lambda Hyperparameters ===")
best_lam = find_best_lambdas(X_train, Y_train, kp_idx, dx)
print("\nBest Lambda Set Found:")
print(best_lam)


# ============================================================
# 9. Full Training using Best λ
# ============================================================
print("\n=== Full Training with Best Lambda ===")

model = FNO1d().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 1500

for ep in range(epochs):
    opt.zero_grad()
    pred = model(X_train)

    L = torch.mean((pred - Y_train)**2)
    L += best_lam["λ_slope"]  * slope_loss(pred, Y_train, dx)
    L += best_lam["λ_kp"]     * keypoint_loss(pred[0], Y_train[0], kp_idx)
    L += best_lam["λ_freq"]   * frequency_loss(pred[0], Y_train[0])
    L += best_lam["λ_trend"]  * trend_loss(pred[0], Y_train[0])
    L += best_lam["λ_smooth"] * smoothness_loss(pred[0])

    L.backward()
    opt.step()

    if ep % 150 == 0:
        print(f"Epoch {ep}: loss={L.item():.4e}")


# ============================================================
# 10. Evaluate + Save Results
# ============================================================
model.eval()
train_pred = model(X_train).cpu().detach().numpy()[0] * std_e + mean_e
test_pred  = model(X_test).cpu().detach().numpy()[0] * std_e + mean_e

pd.DataFrame({
    "distance_m": x_train,
    "usgs_elev": usgs_train,
    "track_true_elev": true_train,
    "fno_pred_elev": train_pred
}).to_csv(output_path + "train_results.csv", index=False)

pd.DataFrame({
    "distance_m": x_test,
    "usgs_elev": usgs_test,
    "track_true_elev": true_test,
    "fno_pred_elev": test_pred
}).to_csv(output_path + "test_results.csv", index=False)


# save plots
plt.figure(figsize=(12,5))
plt.plot(x_train, usgs_train, alpha=0.4)
plt.plot(x_train, true_train, linewidth=2)
plt.plot(x_train, train_pred, '--')
plt.title("Training Alignment (Clovis)")
plt.savefig(output_path + "train_alignment.png", dpi=300)
plt.close()

plt.figure(figsize=(12,5))
plt.plot(x_test, usgs_test, alpha=0.4)
plt.plot(x_test, true_test, linewidth=2)
plt.plot(x_test, test_pred, '--')
plt.title("Test Alignment (Amarillo)")
plt.savefig(output_path + "test_alignment.png", dpi=300)
plt.close()

print("\nAll results saved!\n")
