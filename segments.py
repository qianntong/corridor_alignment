import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


DATA_PATH = "/Users/qianqiantong/PycharmProjects/corridor_alignment/corridor_data/"
OUTPUT_PATH = "/Users/qianqiantong/PycharmProjects/corridor_alignment/output/"
os.makedirs(OUTPUT_PATH, exist_ok=True)


# ============================================================
# 1. Load + Resample
# ============================================================
def load_corridor(filepath):
    df = pd.read_excel(filepath)
    x = df["total_dist_meters"].values.astype(float)
    y = df["elevation_meters"].values.astype(float)
    return x, y


def uniform_resample(x, y, N=4096):
    x_new = np.linspace(x.min(), x.max(), N)
    f = interp1d(x, y, fill_value="extrapolate")
    y_new = f(x_new)
    return x_new, y_new


# ============================================================
# 2. Segment Corridor → Same Length
# ============================================================
def segment_corridor(x, usgs, track, seg_len_m=20000, overlap_m=5000, N=4096):
    segments_usgs, segments_track = [], []
    start = 0.0
    end_x = x[-1]
    step = seg_len_m - overlap_m

    while start + seg_len_m <= end_x:
        mask = (x >= start) & (x <= start + seg_len_m)
        xx = x[mask]
        uu = usgs[mask]
        tt = track[mask]

        if len(xx) < 10:
            break

        x_uniform = np.linspace(start, start + seg_len_m, N)
        fu = interp1d(xx, uu, fill_value="extrapolate")
        ft = interp1d(xx, tt, fill_value="extrapolate")

        segments_usgs.append(fu(x_uniform))
        segments_track.append(ft(x_uniform))

        start += step

    print(f"Total segments extracted = {len(segments_usgs)}")
    return np.array(segments_usgs), np.array(segments_track)


# ============================================================
# 3. Dataset
# ============================================================
class CorridorDataset(Dataset):
    def __init__(self, usgs_segments, track_segments):
        self.usgs = usgs_segments
        self.track = track_segments
        self.N = usgs_segments.shape[1]
        self.grid = np.linspace(0, 1, self.N)

    def __len__(self):
        return len(self.usgs)

    def __getitem__(self, idx):
        u = self.usgs[idx]
        u = (u - u.mean()) / (u.std() + 1e-6)

        xgrid = self.grid
        inp = np.stack([u, xgrid], axis=-1)

        target = self.track[idx]
        return (
            torch.tensor(inp, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32)
        )


# ============================================================
# 4. Stronger FNO Model (Enhanced Stability)
# ============================================================
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_c = in_channels
        self.out_c = out_channels
        self.modes = modes
        scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x):
        B, C, N = x.shape
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(B, self.out_c, N//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes] = torch.einsum(
            "bix, iox -> box",
            x_ft[:, :, :self.modes],
            self.weights
        )
        return torch.fft.irfft(out_ft, n=N)


class FNO1d(nn.Module):
    def __init__(self, modes=64, width=128, layers=6):
        super().__init__()
        self.width = width
        self.fc0 = nn.Linear(2, width)

        self.convs = nn.ModuleList([SpectralConv1d(width, width, modes) for _ in range(layers)])
        self.ws = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(layers)])

        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(layers)])

        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, 1)

    def forward(self, x):
        B, N, _ = x.shape
        x = self.fc0(x).permute(0, 2, 1)

        for conv, w, ln in zip(self.convs, self.ws, self.norms):
            x1 = conv(x)
            x2 = w(x)
            x = x + F.gelu(x1 + x2)   # residual connection
            x = ln(x.permute(0, 2, 1)).permute(0, 2, 1)

        x = x.permute(0, 2, 1)
        x = F.gelu(self.fc1(x))
        return self.fc2(x).squeeze(-1)


# ============================================================
# 5. Trend-aware Loss with Dynamic λ
# ============================================================
def compute_trend(y):
    return y[:, 1:] - y[:, :-1]


def loss_trend(pred, target, lam):
    mse = F.mse_loss(pred, target)
    tp = compute_trend(pred)
    tt = compute_trend(target)
    trend_loss = F.mse_loss(tp, tt)
    return mse + lam * trend_loss, mse.item(), trend_loss.item()


# ============================================================
# 6. Training Loop
# ============================================================
def train_fno(model, loader, epochs=300, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    lam = 10.0  # trend regularization weight

    for ep in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            pred = model(x)
            lam = max(0.1, lam * 0.999)   # slowly decrease λ

            loss, mse, tl = loss_trend(pred, y, lam)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        if ep % 20 == 0:
            print(f"[Epoch {ep}] Loss={loss.item():.4f} MSE={mse:.4f} Trend={tl:.4f} λ={lam:.3f}")

    return model


# ============================================================
# 7. Save CSV
# ============================================================
def save_csv(path, x, usgs, pred, true):
    df = pd.DataFrame({"x": x, "usgs": usgs, "pred": pred, "true": true})
    df.to_csv(path, index=False)
    print(f"[+] Saved CSV: {path}")


# ============================================================
# 8. Plotting
# ============================================================
def plot_training(xgrid, usgs, pred, true):
    plt.figure(figsize=(12, 5))
    plt.plot(xgrid, true, label="TrackChart", linewidth=2)
    plt.plot(xgrid, usgs, label="USGS", alpha=0.7)
    plt.plot(xgrid, pred, label="Prediction", linewidth=2)
    plt.legend()
    plt.grid(alpha=0.3)
    fp = OUTPUT_PATH + "plot_training_clovis.png"
    plt.savefig(fp, dpi=300)
    plt.close()
    print("[+] Saved:", fp)


def plot_test(x, usgs, pred, true):
    plt.figure(figsize=(14, 5))
    plt.plot(x, true, label="TrackChart", linewidth=2)
    plt.plot(x, usgs, label="USGS", alpha=0.7)
    plt.plot(x, pred, label="Prediction", linewidth=2)
    plt.legend()
    plt.grid(alpha=0.3)
    fp = OUTPUT_PATH + "plot_test_amarillo.png"
    plt.savefig(fp, dpi=300)
    plt.close()
    print("Saved:", fp)


# ============================================================
# 9. Main Pipeline
# ============================================================
if __name__ == "__main__":

    # -------- Load Clovis ----------
    x1, u1 = load_corridor(DATA_PATH + "usgs_elevation_Clovis-Flagstaff_grade_data.xlsx")
    x2, t1 = load_corridor(DATA_PATH + "tt_elevation_Clovis-Flagstaff_grade_data.xlsx")

    x_uni, u_uni = uniform_resample(x1, u1)
    _, t_uni = uniform_resample(x2, t1)

    # -------- Segment ----------
    seg_usgs, seg_track = segment_corridor(
        x_uni, u_uni, t_uni, seg_len_m=20000, overlap_m=5000
    )

    # -------- Dataset ----------
    dataset = CorridorDataset(seg_usgs, seg_track)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # -------- Train Model ----------
    model = FNO1d().to(device)
    model = train_fno(model, loader, epochs=300)

    # -------- Save Training Prediction ----------
    model.eval()
    with torch.no_grad():
        s_usgs = seg_usgs[0]
        xgrid = np.linspace(0, 1, 4096)
        inp = np.stack([s_usgs, xgrid], axis=-1)
        pred = model(torch.tensor(inp, dtype=torch.float32).unsqueeze(0).to(device)).cpu().numpy()[0]

    save_csv(OUTPUT_PATH + "training_prediction_clovis.csv", xgrid, s_usgs, pred, seg_track[0])
    plot_training(xgrid, s_usgs, pred, seg_track[0])

    # -------- Test on Amarillo ----------
    xa, ua = load_corridor(DATA_PATH + "usgs_elevation_Amarillo-FortWorth_grade_data.xlsx")
    xt, ta = load_corridor(DATA_PATH + "tt_elevation_Amarillo-FortWorth_grade_data.xlsx")

    x2_uni, ua_uni = uniform_resample(xa, ua)
    _, ta_uni = uniform_resample(xt, ta)

    inp2 = np.stack([ua_uni, np.linspace(0, 1, 4096)], axis=-1)
    pred2 = model(torch.tensor(inp2, dtype=torch.float32).unsqueeze(0).to(device)).cpu().numpy()[0]

    save_csv(OUTPUT_PATH + "test_prediction_amarillo.csv", x2_uni, ua_uni, pred2, ta_uni)
    plot_test(x2_uni, ua_uni, pred2, ta_uni)

    print("\nTraining & Testing Complete.")
    print("Outputs saved to:", OUTPUT_PATH)
