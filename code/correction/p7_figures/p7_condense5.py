r"""P7 page-budget pass 5: appendix wording (status-table rows, notes)."""
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


rep(r"Completed: corrected M valid in 249/254 cells (3 nonconvergence, 1 unidentified, 1 structural non-existence); matched comparison reported on 249.",
    r"Completed: corrected M valid in 249/254 cells; matched comparison on these 249.")
rep(r"Completed: valid in 252/254 cells; preferred over M by AIC/BIC/LRT in all 249 jointly valid cells.",
    r"Completed: valid in 252/254 cells; preferred over M in all 249 jointly valid cells.")
rep(r"Completed: valid in 238/254 cells with intercept-only inflation and 237/254 with promotion-dependent inflation.",
    r"Completed: valid in 238/254 (intercept-only) and 237/254 (promotion-dependent) cells.")
rep(r"Completed: fixed $\pi=0.95/0.85/0.75$ valid in 245/247/248 cells; estimated $\hat\pi$ in 248/254. Similar gaps; M preferred by AIC/BIC/LRT in every jointly valid cell.",
    r"Completed: valid in 245/247/248 (fixed $\pi$) and 248 ($\hat\pi$) of 254 cells; similar gaps; M preferred.")
rep(r"Completed on 20 selected cells: jointly valid in 20/20 after the correction; all-censored store--days counted, not fitted.",
    r"Completed: jointly valid in 20/20 selected cells.")
rep(r"""Completed (E2, \S\ref{sec:3-10}--\S\ref{sec:4-15}; reconstructed inputs): M, NB2, and MIX robust to distributional misspecification and label noise (A0--A3). Under A4 with $\rho>0$ the censored-Poisson MLE does not exist in most datasets (M invalid in 77.6\%/90.5\% of attempts); on valid fits 5 of 12 Holm-adjusted M-vs-naive/G tests are significant, in both directions. Scoped to the simulated construction.""",
    r"""Completed (E2, \S\ref{sec:3-10}; reconstructed inputs): robust under A0--A3; under A4 with $\rho>0$ the MLE usually does not exist (M invalid in 77.6\%/90.5\%); on valid fits 5 of 12 Holm tests significant, in both directions.""")
rep(r"which motivated checking NB2 in the first place; NB2 is valid in 252/254 cells and is the data-preferred family (below).",
    r"which motivated checking NB2.")
rep(r" The NB2-reference depth gap is larger partly because the reference distribution also changes.", "")
rep(r"""Pilot criteria were fixed before the main analysis (unregistered). The E2 simulation (\S\ref{sec:3-10}) is reproduced by \texttt{main/11\_e2\_run\_all.py} (seed 42; city-cluster bootstrap $n_{\text{boot}}=4000$); the A4 fit-level error rates ($n=444/200$) by a separate script in the correction package. The real-data recovery-model sensitivity check (\S\ref{sec:3-9}) is reproduced by \texttt{main/07\_pending\_3.py} $\to$ \texttt{08\_pending\_final.py} $\to$ \texttt{10\_fix\_posthoc.py}; model-preference counts use jointly valid cells.""",
    r"""Pilot criteria were fixed before the main analysis (unregistered). E2 (\S\ref{sec:3-10}) is reproduced by \texttt{main/11\_e2\_run\_all.py} (seed 42; bootstrap $n_{\text{boot}}=4000$), the recovery-model sensitivity check (\S\ref{sec:3-9}) by \texttt{main/07\_pending\_3.py} $\to$ \texttt{08\_pending\_final.py} $\to$ \texttt{10\_fix\_posthoc.py}.""")
rep(r"""The solver contract was frozen before any production fit, and 26/26 numerical checks pass. On the 249 valid empirical cells, corrected and earlier M coefficients differ by a median of $2.1\times10^{-6}$ (promotion) and $1.5\times10^{-5}$ (depth).""",
    r"""The solver contract was frozen before any production fit; 26/26 numerical checks pass. On the 249 valid cells corrected and earlier M coefficients differ by a median of $2.1\times10^{-6}$ (promotion) and $1.5\times10^{-5}$ (depth).""")
rep(r"""This shows that the observed-versus-recovered gap is not specific to the Poisson recovery family, not that any recovery is closer to true demand, and it is not a matched alternative-family comparison (observed-sales NB2 fits are not recorded). We report both estimates conditional on the recovery model (\S\ref{sec:4-13}).""",
    r"""This shows that the gap is not specific to the Poisson recovery family, not that any recovery is closer to true demand; it is not a matched alternative-family comparison (\S\ref{sec:4-13}).""")
open(P, "w").write(s)
print("OK")
