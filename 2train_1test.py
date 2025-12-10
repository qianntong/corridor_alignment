import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os, time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Paths
# ============================================================
data_path = "/Users/qianqiantong/PycharmProjects/corridor_alignment/corridor_data/"
output_path = "/Users/qianqiantong/PycharmProjects/corridor_alignment/output/"
os.makedirs(output_path, exist_ok=True)

# ============================================================
# 1. Load Corridor with slope + curvature
# ============================================================
def load_corridor(excel_usgs, excel_track, N=4096):
    df1 = pd.read_excel(excel_usgs)
    df2 = pd.read_excel(excel_track)

    x1, e1 = df1["total_dist_meters"].values, df1["elevation_meters"].values
    x2, e2 = df2["total_dist_meters"].values, df2["elevation_meters"].values

    L = min(x1.max(), x2.max())
    x_uni = np.linspace(0, L, N)

    f1 = interp1d(x1, e1, kind="linear", fill_value="extrapolate")
    f2 = interp1d(x2, e2, kind="linear", fill_value="extrapolate")

    usgs = f1(x_uni)
    track = f2(x_uni)

    dx = x_uni[1] - x_uni[0]
    slope = np.gradient(usgs, dx)
    curvature = np.gradient(slope, dx)

    return x_uni, usgs, track, slope, curvature

# ============================================================
# Normalize across corridors
# ============================================================
def normalize_multi(arr_list):
    all_values = np.concatenate(arr_list)
    mean = all_values.mean()
    std = all_values.std() + 1e-6
    return [(arr - mean)/std for arr in arr_list], mean, std

# ============================================================
# 2. FNO Model with safe Fourier
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
        B, C, F = x_ft.shape

        m = min(self.modes, F)
        out_ft = torch.zeros(B, self.weights.size(1), F,
                             dtype=torch.cfloat, device=x.device)

        out_ft[:, :, :m] = torch.einsum("bix,iox->box",
                                        x_ft[:, :, :m],
                                        self.weights[:, :, :m])

        return torch.fft.irfft(out_ft, n=x.shape[-1])


class FNO1d(nn.Module):
    def __init__(self, modes=16, width=64, layers=4):
        super().__init__()
        self.fc0 = nn.Linear(4, width)

        self.fourier_layers = nn.ModuleList(
            [SpectralConv1d(width, width, modes) for _ in range(layers)]
        )
        self.conv_layers = nn.ModuleList(
            [nn.Conv1d(width, width, 1) for _ in range(layers)]
        )

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc0(x)      # (B,N,4) -> (B,N,64)
        x = x.permute(0,2,1) # (B,64,N)

        for f, c in zip(self.fourier_layers[:-1], self.conv_layers[:-1]):
            x = F.gelu(f(x) + c(x))

        x = self.fourier_layers[-1](x) + self.conv_layers[-1](x)
        x = x.permute(0,2,1)

        x = F.gelu(self.fc1(x))
        return self.fc2(x).squeeze(-1)

# ============================================================
# 3. Loss Functions
# ============================================================
def slope_loss(pred, true, dx):
    return ((torch.diff(pred)/dx - torch.diff(true)/dx)**2).mean()

def smoothness_loss(pred):
    d2 = torch.diff(pred, n=2)
    return (d2**2).mean()

# ============================================================
# 4. Load Training Corridors
# ============================================================
train_corridors = [
    ("Clovis",
     data_path+"usgs_elevation_Clovis-Flagstaff_grade_data.xlsx",
     data_path+"tt_elevation_Clovis-Flagstaff_grade_data.xlsx"),

    ("Amarillo",
     data_path+"usgs_elevation_Amarillo-FortWorth_grade_data.xlsx",
     data_path+"tt_elevation_Amarillo-FortWorth_grade_data.xlsx")
]

test_corridor = (
    "Barstow",
    data_path+"usgs_elevation_Barstow-LongBeach_grade_data.xlsx",
    data_path+"tt_elevation_Barstow-LongBeach_grade_data.xlsx"
)

# ---- Load raw data ----
X_raw_list, Y_raw_list, slopes_raw, curv_raw, x_raw = [], [], [], [], []
for _, usgs_path, track_path in train_corridors:
    x, usgs, track, slope, curv = load_corridor(usgs_path, track_path)
    X_raw_list.append(usgs)
    Y_raw_list.append(track)
    slopes_raw.append(slope)
    curv_raw.append(curv)
    x_raw.append(x)

(X_norm_list, slopes_norm_list, curv_norm_list), mean_e, std_e = normalize_multi(
    [np.concatenate(X_raw_list), np.concatenate(slopes_raw), np.concatenate(curv_raw)]
)

# split back by corridor
split_sizes = [len(arr) for arr in X_raw_list]
def split_norm(norm_arr, sizes):
    out, idx = [], 0
    for s in sizes:
        out.append(norm_arr[idx:idx+s])
        idx += s
    return out

X_norm_list      = split_norm(X_norm_list, split_sizes)
slopes_norm_list = split_norm(slopes_norm_list, split_sizes)
curv_norm_list   = split_norm(curv_norm_list, split_sizes)

# ---- build training tensor ----
X_train_list, Y_train_list = [], []
for i in range(len(X_raw_list)):
    xnorm = x_raw[i] / x_raw[i].max()
    X_train_list.append(
        np.stack([X_norm_list[i], slopes_norm_list[i], curv_norm_list[i], xnorm], axis=-1)
    )
    Y_train_list.append((Y_raw_list[i] - mean_e) / std_e)

X_train = torch.tensor(np.stack(X_train_list), dtype=torch.float32).to(device)
Y_train = torch.tensor(np.stack(Y_train_list), dtype=torch.float32).to(device)

dx = x_raw[0][1] - x_raw[0][0]

# ============================================================
# 5. Load Test Corridor
# ============================================================
x_test, usgs_test, true_test, s_test, c_test = load_corridor(
    test_corridor[1], test_corridor[2]
)

usgs_test_n = (usgs_test - mean_e) / std_e
s_test_n    = (s_test   - mean_e) / std_e
c_test_n    = (c_test   - mean_e) / std_e

X_test = torch.tensor(np.stack([
    usgs_test_n, s_test_n, c_test_n, x_test/x_test.max()
], axis=-1), dtype=torch.float32).unsqueeze(0).to(device)

Y_test = torch.tensor((true_test - mean_e)/std_e,
                      dtype=torch.float32).unsqueeze(0).to(device)

# ============================================================
# 6. AUTO TUNING λ (slope, smooth)
# ============================================================
lambda_candidates = []
lambda_scores = []

λ_slope_list  = [0.01, 0.05, 0.1, 0.2]
λ_smooth_list = [0.001, 0.005, 0.01]

def evaluate_lambda(λ_slope, λ_smooth):
    model = FNO1d().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=5e-4)

    # Short training (fast evaluation)
    for _ in range(40):
        opt.zero_grad()
        pred = model(X_train)

        loss = ((pred - Y_train)**2).mean()
        loss += λ_slope * slope_loss(pred, Y_train, dx)
        loss += λ_smooth * smoothness_loss(pred)

        loss.backward()
        opt.step()

    with torch.no_grad():
        final_pred = model(X_train)
        mse = ((final_pred - Y_train)**2).mean().item()
    return mse

print("\n=== Auto λ Search ===")
for λs in λ_slope_list:
    for λc in λ_smooth_list:
        score = evaluate_lambda(λs, λc)
        lambda_candidates.append((λs, λc))
        lambda_scores.append(score)
        print(f"λ_slope={λs}, λ_smooth={λc} → score={score:.6f}")

best_idx = np.argmin(lambda_scores)
λ_slope_best, λ_smooth_best = lambda_candidates[best_idx]

print("\nBest λ found:")
print("λ_slope  =", λ_slope_best)
print("λ_smooth =", λ_smooth_best)

# plot lambda search result
plt.figure(figsize=(7,5))
plt.scatter(range(len(lambda_scores)), lambda_scores, c='blue')
plt.title("Lambda Search Results")
plt.xlabel("Trial Index")
plt.ylabel("Score (lower=better)")
plt.savefig(output_path + "lambda_search_curve.png", dpi=300)
plt.close()

# ============================================================
# 7. Training with best λ
# ============================================================
model = FNO1d().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 2000
loss_curve = []

for ep in range(epochs):
    opt.zero_grad()
    pred = model(X_train)

    loss = ((pred - Y_train)**2).mean()
    loss += λ_slope_best * slope_loss(pred, Y_train, dx)
    loss += λ_smooth_best * smoothness_loss(pred)

    loss.backward()
    opt.step()
    loss_curve.append(loss.item())

    if ep % 200 == 0:
        print(f"[{ep}] loss = {loss.item():.6f}")

# ============================================================
# 8. Evaluate, Save Results
# ============================================================
model.eval()
train_pred = model(X_train).cpu().detach().numpy() * std_e + mean_e
test_pred = model(X_test).cpu().detach().numpy()[0] * std_e + mean_e

# save lambda scan
pd.DataFrame({
    "λ_slope": [x[0] for x in lambda_candidates],
    "λ_smooth": [x[1] for x in lambda_candidates],
    "score": lambda_scores
}).to_csv(output_path + "lambda_scan.csv", index=False)

# save loss
plt.figure(figsize=(10,4))
plt.plot(loss_curve)
plt.title("Training Loss with Best λ")
plt.savefig(output_path + "loss_curve.png", dpi=300)
plt.close()

print("\nAll results saved to:", output_path)
