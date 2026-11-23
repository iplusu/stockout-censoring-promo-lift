r"""P7 page-budget pass 7: appendix wording."""
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


rep(r"""A dispersion diagnostic on uncensored days shows moderate overdispersion (Pearson $\chi^{2}$/df median 3.22), which motivated checking NB2.

""", "")
rep(r""" Earlier converged-subset results (16 NB2 and 9 ZIP cells) came from non-valid fits and are superseded.""", r""" Earlier converged-subset results (16 NB2, 9 ZIP cells) are superseded.""")
rep(r"""The 25.9\%/52.5\% medians \emph{survive} an alternative if, on the cells where it is valid, the median gap is at least 20\%, the share above 20\% is at least 50\%, and the cluster-interval lower bound exceeds 20\%. Every alternative satisfies the rule for both coefficients, which revises the earlier statement that NB2/ZIP convergence was too sparse to apply it. This shows that the gap is not specific to the Poisson recovery family, not that any recovery is closer to true demand; it is not a matched alternative-family comparison (\S\ref{sec:4-13}).""",
    r"""The 25.9\%/52.5\% medians \emph{survive} an alternative if, on its valid cells, the median gap is at least 20\%, the share above 20\% at least 50\%, and the cluster-interval lower bound above 20\%. Every alternative satisfies the rule for both coefficients (revising the earlier statement that NB2/ZIP convergence was too sparse to apply it): the gap is not specific to the Poisson recovery family, which does not make any recovery closer to true demand (\S\ref{sec:4-13}).""")
rep(r"""E2 is a documented reconstruction: keys, true coefficients and anchors are identical for all 35,100 datasets and generated data for 28,985; exact bit replay of the rest was not possible because the generator calls the earlier M fit. A correction package (\texttt{repair\_log.md}, \texttt{fit\_diagnostics.csv} with every attempted fit, \texttt{before\_after.csv} with each published and corrected value) is in the replication archive.""",
    r"""E2 is a documented reconstruction (keys and true coefficients identical for all 35,100 datasets, generated data for 28,985; the generator calls the earlier M fit, so exact bit replay was not possible). The replication archive contains a correction package: \texttt{repair\_log.md}, \texttt{fit\_diagnostics.csv} (every attempted fit) and \texttt{before\_after.csv} (each published and corrected value).""")
rep(r"""\textbf{Model preference (run).} On jointly valid cells, M beats the mixture-censoring specifications by AIC/BIC/LRT in every cell (242/245/244 at $\pi=0.95/0.85/0.75$, 244 with $\hat\pi$; median $\Delta$AIC $+226$/$+718$/$+1{,}277$/$+3.4$).""",
    r"""\textbf{Model preference (run).} On jointly valid cells M beats the mixture-censoring specifications by AIC/BIC/LRT in every cell (242/245/244 at fixed $\pi$, 244 with $\hat\pi$; median $\Delta$AIC $+226$/$+718$/$+1{,}277$/$+3.4$).""")
open(P, "w").write(s)
print("OK")
