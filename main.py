import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Paths
# ============================================================
data_path = "/Users/qianqiantong/PycharmProjects/corridor_alignment/corridor_data/"
output_path = "/Users/qianqiantong/PycharmProjects/corridor_alignment/output/"
os.makedirs(output_path, exist_ok=True)


# ============================================================
# 1. Load + Interpolate
# ============================================================
def load_and_interpolate(excel_usgs, excel_track, N=4096):
    df1 = pd.read_excel(excel_usgs)
    df2 = pd.read_excel(excel_track)

    x1, e1 = df1["total_dist_meters"].values, df1["elevation_meters"].values
    x2, e2 = df2["total_dist_meters"].values, df2["elevation_meters"].values

    L = min(x1.max(), x2.max())
    x_uniform = np.linspace(0, L, N)

    f1 = interp1d(x1, e1, fill_value="extrapolate")
    f2 = interp1d(x2, e2, fill_value="extrapolate")

    return x_uniform, f1(x_uniform), f2(x_uniform)


# ============================================================
# 2. Feature extraction: smoothing + slope + curvature
# ============================================================
def compute_features(elev, x, sigma=5):
    elev_s = gaussian_filter1d(elev, sigma=sigma)
    dx = x[1] - x[0]
    slope = np.gradient(elev_s, dx)
    curvature = np.gradient(slope, dx)
    return elev_s, slope, curvature


# ============================================================
# 3. FNO (safe spectral conv)
# ============================================================
class SpectralConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, modes):
        super().__init__()
        self.modes = modes
        self.scale = 1/(in_ch * out_ch)
        self.weights = nn.Parameter(
            self.scale * torch.rand(in_ch, out_ch, modes, dtype=torch.cfloat)
        )

    def forward(self, x):
        B, C, N = x.shape
        x_ft = torch.fft.rfft(x)                 # [B, C, N//2+1]
        freq_count = x_ft.shape[-1]

        m = min(self.modes, freq_count)
        out_ft = torch.zeros(B, self.weights.size(1), freq_count,
                             dtype=torch.cfloat, device=x.device)

        w = self.weights[:, :, :m]
        x_sel = x_ft[:, :, :m]
        out_ft[:, :, :m] = torch.einsum("bcm, com -> bom", x_sel, w)

        return torch.fft.irfft(out_ft, n=N)


class FNO1d(nn.Module):
    def __init__(self, modes=32, width=64, layers=4):
        super().__init__()
        self.fc0 = nn.Linear(4, width)  # 4 input features

        self.f_layers = nn.ModuleList([SpectralConv1d(width, width, modes)
                                       for _ in range(layers)])
        self.c_layers = nn.ModuleList([nn.Conv1d(width, width, 1)
                                       for _ in range(layers)])

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc0(x)               # [B, N, width]
        x = x.permute(0, 2, 1)        # [B, width, N]

        for f, c in zip(self.f_layers, self.c_layers):
            x = F.gelu(f(x) + c(x))

        x = x.permute(0, 2, 1)
        x = F.gelu(self.fc1(x))
        return self.fc2(x).squeeze(-1)


# ============================================================
# 4. Loss functions
# ============================================================
def slope_loss(pred, true, dx):
    return torch.mean((torch.gradient(pred, spacing=(dx,))[0] -
                       torch.gradient(true, spacing=(dx,))[0])**2)

def curvature_loss(pred, true, dx):
    dp = torch.gradient(pred, spacing=(dx,))[0]
    dt = torch.gradient(true, spacing=(dx,))[0]
    cp = torch.gradient(dp, spacing=(dx,))[0]
    ct = torch.gradient(dt, spacing=(dx,))[0]
    return torch.mean((cp - ct)**2)

def smoothness_loss(pred):
    d2 = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]
    return torch.mean(d2**2)


# ============================================================
# 5. Prepare train & test
# ============================================================
N = 4096
x_train, usgs_train, true_train = load_and_interpolate(
    data_path+"usgs_elevation_Barstow-LongBeach_grade_data.xlsx",
    data_path+"tt_elevation_Barstow–LongBeach_grade_data.xlsx",
    N=N
)
x_test, usgs_test, true_test = load_and_interpolate(
    data_path+"usgs_elevation_Barstow–LongBeach_grade_data.xlsx",
    data_path+"tt_elevation_Barstow–LongBeach_grade_data.xlsx",
    N=N
)

# feature extraction
elev_s_train, slope_train, curv_train = compute_features(usgs_train, x_train)
elev_s_test, slope_test, curv_test = compute_features(usgs_test, x_test)

# normalization (use training stats)
mean = true_train.mean()
std = true_train.std()

# normalized features
train_feat = np.stack([
    (elev_s_train - mean) / std,
    slope_train / (np.std(slope_train) + 1e-6),
    curv_train / (np.std(curv_train) + 1e-6),
    x_train / x_train.max(),
], axis=-1)

test_feat = np.stack([
    (elev_s_test - mean) / std,
    slope_test / (np.std(slope_train) + 1e-6),
    curv_test / (np.std(curv_train) + 1e-6),
    x_test / x_test.max(),
], axis=-1)

Y_train = (true_train - mean) / std
Y_test = (true_test - mean) / std

X_train = torch.tensor(train_feat, dtype=torch.float32).unsqueeze(0).to(device)
Y_train = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(0).to(device)
X_test  = torch.tensor(test_feat,  dtype=torch.float32).unsqueeze(0).to(device)
Y_test  = torch.tensor(Y_test,  dtype=torch.float32).unsqueeze(0).to(device)

dx = x_train[1] - x_train[0]


# ============================================================
# 6. Train
# ============================================================
model = FNO1d().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

loss_history = []
epochs = 1000

for ep in range(epochs):
    opt.zero_grad()
    pred = model(X_train)

    L_e = torch.mean((pred - Y_train)**2)
    L_s = slope_loss(pred[0], Y_train[0], dx)
    L_c = curvature_loss(pred[0], Y_train[0], dx)
    L_sm = smoothness_loss(pred)

    loss = L_e + 0.1*L_s + 0.02*L_c + 0.001*L_sm
    loss.backward()
    opt.step()

    loss_history.append(loss.item())

    if ep % 100 == 0:
        print(f"Epoch {ep}: loss = {loss.item():.6f}")


# ============================================================
# 7. Evaluate
# ============================================================
model.eval()
train_pred = model(X_train).detach().cpu().numpy()[0] * std + mean
test_pred  = model(X_test).detach().cpu().numpy()[0] * std + mean


# ============================================================
# 8. Save CSV
# ============================================================
pd.DataFrame({
    "distance_m": x_train,
    "usgs": usgs_train,
    "tc_true": true_train,
    "fno_pred": train_pred
}).to_csv(output_path+"train_results.csv", index=False)

pd.DataFrame({
    "distance_m": x_test,
    "usgs": usgs_test,
    "tc_true": true_test,
    "fno_pred": test_pred
}).to_csv(output_path+"test_results.csv", index=False)


# ============================================================
# 9. Loss plot
# ============================================================
plt.figure(figsize=(7,4))
plt.plot(loss_history)
plt.title("Training Loss")
plt.grid()
plt.savefig(output_path+"training_loss.png", dpi=300)
plt.close()


# ============================================================
# 10. Plot elevation curves
# ============================================================
plt.figure(figsize=(12,5))
plt.plot(x_train, usgs_train, alpha=0.4, label="USGS")
plt.plot(x_train, true_train, label="Track Chart", linewidth=2)
plt.plot(x_train, train_pred, '--', label="FNO Pred")
plt.legend()
plt.grid()
plt.savefig(output_path+"train_alignment.png", dpi=300)
plt.close()

plt.figure(figsize=(12,5))
plt.plot(x_test, usgs_test, alpha=0.4, label="USGS")
plt.plot(x_test, true_test, label="Track Chart", linewidth=2)
plt.plot(x_test, test_pred, '--', label="FNO Pred")
plt.legend()
plt.grid()
plt.savefig(output_path+"test_alignment.png", dpi=300)
plt.close()

print("\nAll results saved!\n")