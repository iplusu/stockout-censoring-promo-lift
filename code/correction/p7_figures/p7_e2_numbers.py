"""P7: E2 arm-A4 numbers for the manuscript (corrected vs original), incl. all-attempt failure
accounting from fit_diagnostics.csv and the fit-level extreme-error shares on valid M fits.
Gate: the 'orig' set must reproduce the published 74.1%/89.3%/34.8%/40.8% (n=2,144/2,280)."""
import os
import numpy as np, pandas as pd

OUT = os.path.expanduser("~/papers/jams-frn50k/correction_20261122/out")


def extreme(setname):
    a = pd.read_parquet(f"{OUT}/p5_e2/{setname}/e2_all2.parquet")
    a = a[(a.arm == "A4_nonignorable") & a.param.isin([0.5, 1.0])]
    ok = a.m_conv.astype(bool) & (a.beta_true_p.abs() >= .05) & (a.beta_true_d.abs() >= .05)
    s = a[ok]
    ep = (s.m_p - s.beta_true_p).abs() / s.beta_true_p.abs()
    ed = (s.m_d - s.beta_true_d).abs() / s.beta_true_d.abs()
    res = {}
    for rho in (0.5, 1.0):
        k = s.param == rho
        res[rho] = (int(k.sum()), ((ep > 1) | (ed > 1))[k].mean(), ((ep > 1) & (ed > 1))[k].mean())
        print(f"  [{setname}] rho={rho}: n={res[rho][0]} either>1 {100*res[rho][1]:.1f}% both>1 {100*res[rho][2]:.1f}%")
    return res


if __name__ == "__main__":
    o = extreme("orig")
    assert o[0.5][0] == 2144 and o[1.0][0] == 2280 and round(100 * o[0.5][1], 1) == 74.1 and round(100 * o[1.0][2], 1) == 40.8
    print("GATE PASS (orig reproduces published fit-level shares)")
    extreme("corrected")

    fd = pd.read_csv(f"{OUT}/fit_diagnostics.csv", low_memory=False)
    e = fd[(fd.source == "E2") & (fd.arm == "A4_nonignorable")]
    print("\nA4 all-attempt status by rho x model (corrected):")
    tab = e.groupby(["param", "model"]).corrected_status.value_counts().unstack(fill_value=0)
    print(tab.to_string())
    n = e.groupby(["param", "model"]).size()
    val = e[e.corrected_status == "valid"].groupby(["param", "model"]).size()
    print("\ninvalid share (all attempts):"); print(((1 - val / n) * 100).round(1).unstack().to_string())
    m = e[e.model == "M"]
    ne = m[m.corrected_status == "structural_nonexistence"]
    print("\nM nonexistence datasets:", len(ne), "of which legacy flagged converged (original shard):",
          int(ne.original_shard_converged.astype(str).eq("True").sum()), "| legacy replay converged:",
          int(ne.legacy_replay_converged.astype(str).eq("True").sum()))
    allm = fd[(fd.source == "E2") & (fd.model == "M")]
    ne_all = allm[allm.corrected_status == "structural_nonexistence"]
    print("all-arm M nonexistence:", len(ne_all), "legacy-converged (orig shard):", int(ne_all.original_shard_converged.astype(str).eq("True").sum()),
          "| replay:", int(ne_all.legacy_replay_converged.astype(str).eq("True").sum()))
    print("median cens share by rho:", m.groupby("param").cens_share.median().round(4).to_dict())
