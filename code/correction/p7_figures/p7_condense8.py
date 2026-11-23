r"""P7 page-budget pass 8: appendix wording."""
import os, re
P = os.path.expanduser("~/papers/jams-frn50k/latex/main.tex")
s = open(P).read()


def rep(old, new):
    global s
    assert s.count(old) == 1, old[:70]
    gone = sorted(set(re.findall(r"\d+(?:[.,]\d+)?", old)) - set(re.findall(r"\d+(?:[.,]\d+)?", new)))
    if gone:
        print(f"[{old[:40]}] numbers dropped: {gone}")
    s = s.replace(old, new)


rep(r"Recovery alternatives were refitted on 254 cells under the corrected solver contract (seed 42); fits that are not valid are reported and never used.",
    r"Alternatives were refitted on 254 cells (seed 42); non-valid fits are never used.")
rep(r"NB2 is data-preferred (249/249 cells, median $\Delta$AIC $-4{,}497$; NB2+mixture 245/247, 221/247, 192/248, 249/249), a specification diagnostic only.",
    r"NB2 is data-preferred (249/249 cells, median $\Delta$AIC $-4{,}497$; NB2+mixture 245/247, 221/247, 192/248, 249/249).")
rep(r"Median $|b_{\text{G}}-b_{\text{alt}}|/|b_{\text{alt}}|$ with Rademacher wild-cluster 95\% intervals (18 cities) and share $\geq$20\%, on the cells where each alternative is valid:",
    r"Median $|b_{\text{G}}-b_{\text{alt}}|/|b_{\text{alt}}|$ [wild-cluster 95\% interval] and share $\geq$20\% on valid cells:")
rep(r"""\item The model-relative gap is negative at 22 of 24 hours (naive below recovered in all but 07:00 and 08:00), i.e. the naive estimate understates the latent promotion effect at most hours, with two marginal exceptions.""",
    r"""\item The model-relative gap is negative at 22 of 24 hours (all but 07:00 and 08:00, both marginal).""")
rep(r"""Pilot criteria were fixed before the main analysis (unregistered). E2 (\S\ref{sec:3-10}) is reproduced by""", r"""Pilot criteria were fixed before the main analysis (unregistered). E2 is reproduced by""")
open(P, "w").write(s)
print("OK")
