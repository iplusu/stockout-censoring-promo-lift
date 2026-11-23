"""P7: hour-of-day numbers for the manuscript from the corrected P4 arrays.
H0 = original (wrong row mask), HA = mask fix, HB = mask fix + corrected M (valid cells only).
Check: H0 aggregated over 254 cells must reproduce the published main/main_hourly.parquet means."""
import glob, os
import numpy as np, pandas as pd

ROOT = os.path.expanduser("~/papers/jams-frn50k")
P4 = f"{ROOT}/correction_20261122/out/p4"
summ = pd.read_csv(f"{ROOT}/correction_20261122/out/p4_hourly_summary.csv")


def load(variant):
    rows = []
    for _, r in summ.iterrows():
        if variant == "HB" and r.corrected_M_status != "valid":
            continue  # no valid corrected M -> no HB array
        z = np.load(f"{P4}/{int(r.city)}_{int(r["cat"])}_hourly.npz")
        rec = z[f"rec_h_{variant}"]
        rows.append((int(r.city), int(r["cat"]), z["naive_h"], rec))
    return rows


def stats(rows, label):
    nv = np.array([r[2] for r in rows]); rc = np.array([r[3] for r in rows])
    n_mean, r_mean = nv.mean(0), rc.mean(0)
    bias = (nv - rc).mean(0)
    order = np.argsort(-np.abs(bias))
    print(f"--- {label}: n_cells={len(rows)}")
    print(" hours naive<rec:", int((n_mean < r_mean).sum()), " exceptions:", [(h, round(n_mean[h], 4), round(r_mean[h], 4)) for h in range(24) if n_mean[h] >= r_mean[h]])
    print(" bias at 0,6,12,18,23:", [round(bias[h], 3) for h in (0, 6, 12, 18, 23)])
    print(" top5 |bias| hours:", order[:5].tolist(), " peak rec hour:", int(np.argmax(r_mean)), " max|bias| hour", int(order[0]), round(bias[order[0]], 3))
    print(" day totals naive %.2f rec %.2f  understate %.1f%%" % (n_mean.sum(), r_mean.sum(), 100 * (1 - n_mean.sum() / r_mean.sum())))
    return n_mean, r_mean, bias


if __name__ == "__main__":
    pub = pd.read_parquet(f"{ROOT}/main/main_hourly.parquet")
    n0, r0, b0 = stats(load("H0"), "H0 original mask (254)")
    assert np.allclose(r0, pub[[f"rec_h{h}" for h in range(24)]].mean().values, atol=1e-4), "H0 does not reproduce published"
    assert np.allclose(n0, pub[[f"naive_h{h}" for h in range(24)]].mean().values, atol=1e-6)
    print("H0 reproduces published main_hourly.parquet means: OK")
    stats(load("HA"), "HA mask fix (254)")
    stats(load("HB"), "HB mask fix + corrected M (249)")
    for c in ("peak_rec", "peak_bias"):
        print(c, "changes H0->HA:", int((summ[f"{c}_H0"] != summ[f"{c}_HA"]).sum()), "/", len(summ))
    v = summ[summ.corrected_M_status == "valid"]
    print("peak_rec changes H0->HB on valid:", int((v.peak_rec_H0 != v.peak_rec_HB).sum()), "/", len(v))
    print("modal peak rec H0/HA/HB:", summ.peak_rec_H0.mode().tolist(), summ.peak_rec_HA.mode().tolist(), v.peak_rec_HB.mode().tolist())
