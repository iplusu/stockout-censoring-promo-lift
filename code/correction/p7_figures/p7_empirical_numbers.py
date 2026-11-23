"""P7: every M-based empirical number quoted in main.tex, recomputed from the corrected sandbox
outputs (valid corrected M only), with the same function applied to the original files as a gate."""
import os, json
import numpy as np, pandas as pd

ROOT = os.path.expanduser("~/papers/jams-frn50k")
ORIG, CORR = f"{ROOT}/main", f"{ROOT}/correction_20261122/sandbox/main"
pd.set_option("display.width", 220)


def report(D, tag):
    print(f"\n################ {tag}")
    mc = pd.read_csv(f"{D}/main_cells.csv"); mc = mc[mc.status == "ok"]
    r = mc.bias_ratio.dropna().abs()
    print("P1a n", len(r), "median %.4f" % r.median(), "p10/25/75/90", [round(r.quantile(q), 3) for q in (.1, .25, .75, .9)],
          "mean %.3f" % r.mean(), "ge20 %.4f ge50 %.4f gt100 %.4f" % ((r >= .2).mean(), (r >= .5).mean(), (r > 1).mean()))
    v = mc.dropna(subset=["recA"])
    nA, rA = v.naiveA, v.recA
    att = ((np.sign(nA) == np.sign(rA)) & (nA.abs() < rA.abs()))
    exa = ((np.sign(nA) == np.sign(rA)) & (nA.abs() > rA.abs()))
    flip = np.sign(nA) != np.sign(rA)
    print("direction n", len(v), "atten", int(att.sum()), "(pos", int((att & (nA > 0)).sum()), "neg", int((att & (nA < 0)).sum()), ") exag", int(exa.sum()),
          "flip", int(flip.sum()), "(pos->neg", int((flip & (nA > 0)).sum()), ")", "neg naive share %.4f" % (mc.naiveA < 0).mean(), "neg naive on valid %.4f" % (nA < 0).mean())

    dc = pd.read_csv(f"{D}/pending_table4_commondenom.csv")
    dc = dc.dropna(subset=["beta_rec_depth"])
    print("commondenom n", len(dc))
    for comp, col in [("unmatched", "cd_attr"), ("omission", "cd_omission"), ("dist", "cd_dist"), ("matched", "cd_pure_cens")]:
        for k in ("depth", "promo"):
            x = dc[f"{col}_{k}"]
            print(f"  {comp:9s} {k}: med|c| {x.abs().median():.4f} signed {x.median():+.4f} frac>=.2 {(x.abs() >= .2).mean():.4f}")
    print("  max identity resid", float(np.nanmax(np.abs(dc[["resid_depth", "resid_promo"]].values))))
    for k in ("depth", "promo"):
        u = dc[f"cd_attr_{k}"].abs() >= .2; m = dc[f"cd_pure_cens_{k}"].abs() >= .2
        print(f"  reclass {k}: both {int((u&m).sum())} unm-only {int((u&~m).sum())} mat-only {int((~u&m).sum())} neither {int((~u&~m).sum())} changes {int((u^m).sum())}")
        print(f"  |b_M|<0.05 {k}: {int((dc[f'beta_rec_{k}'].abs() < .05).sum())}")
        fl = (np.sign(dc[f"beta_glm_{k}"]) != np.sign(dc[f"beta_rec_{k}"])).sum()
        fo = (np.sign(dc[f"beta_naive_reduced_{k}"]) != np.sign(dc[f"beta_rec_{k}"])).sum()
        print(f"  signflip {k}: matched {fl}/{len(dc)} = {fl/len(dc):.4f}; orig {fo}/{len(dc)} = {fo/len(dc):.4f}")
    print("  promo<0 naive OLS %.4f  recovered %.4f" % ((dc.beta_naive_reduced_promo < 0).mean(), (dc.beta_rec_promo < 0).mean()))
    m2 = mc.merge(dc[["city", "cat", "cd_attr_depth"]], on=["city", "cat"])
    dd = pd.read_csv(f"{D}/decomp_cells.csv").dropna(subset=["beta_rec_depth"])
    m3 = mc.merge(dd[["city", "cat", "bias_pct_orig_depth"]], on=["city", "cat"]).dropna(subset=["bias_ratio", "bias_pct_orig_depth"])
    print("  corr(|bias_pct_orig_depth|, P1a ratio) = %.4f n=%d" % (np.corrcoef(m3.bias_pct_orig_depth.abs(), m3.bias_ratio)[0, 1], len(m3)))

    do = pd.read_csv(f"{D}/pending_decomp_order.csv")
    for k in ("depth", "promo"):
        ok = do[f"mred_{k}_converged"].astype(str) == "True"
        sub = do[ok].dropna(subset=[f"cens_B_{k}"])
        print(f"order {k}: reduced-spec valid {int(ok.sum())}/{len(do)}; n_used {len(sub)}; censB med {sub[f'cens_B_{k}'].abs().median():.4f} omitB {sub[f'omit_B_{k}'].abs().median():.4f} max resid {np.nanmax(np.abs(sub[f'resid_B_{k}'])):.2e}")

    fv = pd.read_csv(f"{D}/pending_fe_vif.csv").dropna(subset=["b_M_depth"])
    print("VIF n", len(fv), "med promo %.2f depth %.2f; depth>10 %.4f; centering max|d| promo %.2e depth %.2e" % (
        fv.vif_promo.median(), fv.vif_depth.median(), (fv.vif_depth > 10).mean(), fv.d_promo_c.abs().max(), fv.d_depth_c.abs().max()))
    print("  product FE converged", int((fv.fe_converged.astype(str) == "True").sum()), "median n_groups", fv.fe_n_groups.median())

    L = pd.read_csv(f"{D}/discount_scenario_L.csv"); print(L[["d_label", "d", "n_cells_in_range", "median_diff_pp"]].to_string(index=False))
    q = pd.read_csv(f"{D}/pending_final_dose_wildci.csv"); q["ci_lo"] = q.gap + q.lo; q["ci_hi"] = q.gap + q.hi
    print(q.round(3).to_string(index=False), "\n  total n", q.n.sum())
    pa = pd.read_csv(f"{D}/pending_altmodels.csv")
    print("altmodels rows", len(pa), "m_converged", (pa.m_converged.astype(str) == "True").sum())


if __name__ == "__main__":
    report(ORIG, "ORIGINAL (gate: must match published)")
    report(CORR, "CORRECTED")
