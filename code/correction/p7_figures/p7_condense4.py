r"""P7 page-budget pass 4 (wording only)."""
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


rep(r"""G (observed-sales Poisson) and M (censored Poisson, which ignores the dependence by design) are fitted under the validity rule of Appendix~\ref{app:repro}. The loss $A$ is the absolute error of the predicted lift $\exp(\hat\beta_P+d\hat\beta_d)-1$ in percentage points (pp), averaged over $d\in\{0.05,0.15,0.30\}$; $\Delta_{\eta q}$ is the median within-ID difference $B=A_{\text{M}}-A_{\text{G}}$, and the primary statistic $\Theta_{0.60}$ is the median within-ID change in $B$ from $\eta=0$ to $\eta=0.6$ at $q=0.60$, with 95\% Monte Carlo intervals from 2,000 joint bootstrap resamples of IDs. Design detail, failure and information diagnostics (Table A8), the oracle summary and descriptive outputs are in Online Resource 1.""",
    r"""G (observed-sales Poisson) and M (censored Poisson, ignoring the dependence by design) use the validity rule of Appendix~\ref{app:repro}. The loss $A$ is the absolute error of the predicted lift $\exp(\hat\beta_P+d\hat\beta_d)-1$ in percentage points (pp), averaged over $d\in\{0.05,0.15,0.30\}$; $\Delta_{\eta q}$ is the median within-ID difference $B=A_{\text{M}}-A_{\text{G}}$, and the primary statistic $\Theta_{0.60}$ is the median within-ID change in $B$ from $\eta=0$ to $\eta=0.6$ at $q=0.60$ (95\% Monte Carlo intervals, 2,000 joint bootstrap resamples of IDs). Design detail, diagnostics (Table A8) and the oracle summary are in Online Resource 1.""")

rep(r"""although the lift error still favors M. The oracle's median lift loss is 1.03 pp. These results cover""", r"""although the lift error still favors M. These results cover""")

rep(r"""\textbf{Scope, stated precisely.} This is a failure under one generative construction (a shared latent shock linking demand and capacity), not a general law that ``all non-ignorable censoring breaks M.'' The per-dataset censoring shares resolve the earlier open question whether the extreme A4 errors reflected bias or near-total censoring: they reflect near-total censoring and non-existence. Not established are""",
    r"""\textbf{Scope, stated precisely.} This is a failure under one generative construction (a shared latent shock linking demand and capacity), not a general law. The per-dataset censoring shares resolve the earlier open question about the extreme A4 errors: they reflect near-total censoring and non-existence. Not established are""")

rep(r"""\textbf{Resulting recommendation.} Because no validated pre-test for non-ignorable censoring is presented here, we do not prescribe withholding M. Instead, report the naive/G and M-corrected estimates side by side (dual reporting) whenever ignorability has not been independently verified; confirm that M has a finite, valid maximum before reading it, and report a non-existent or non-valid fit as such rather than substituting a number; and treat divergence between the two estimates as the diagnostic signal. The absence of a specific reason to suspect non-ignorable stockout timing (e.g., promotions timed to expected demand, supply shortages correlated with anticipated spikes) does not verify the assumption. This is narrower than \S\ref{sec:3-8}'s original unconditional-substitution claim: robustness to the misspecifications tested in A0--A3 stands, but it does not extend to non-ignorable censoring (A4), where ignorability is an unverified assumption that dual reporting is designed to surface, not assume away.""",
    r"""\textbf{Resulting recommendation.} No validated pre-test for non-ignorable censoring is presented here, so we do not prescribe withholding M. Instead, report the naive/G and M-corrected estimates side by side whenever ignorability has not been independently verified; confirm that M has a finite, valid maximum before reading it, reporting a non-existent or non-valid fit as such; and treat divergence between the two estimates as the diagnostic signal. The absence of a specific reason to suspect non-ignorable stockout timing (e.g., promotions timed to expected demand) does not verify the assumption. This is narrower than \S\ref{sec:3-8}'s original unconditional-substitution claim: robustness to the A0--A3 misspecifications stands but does not extend to non-ignorable censoring (A4).""")

rep(r"""These results condition on a finite MLE, which exists in a selected minority of datasets, and are read with the failure rates above (\S\ref{sec:4-15}).""",
    r"""These results condition on a finite MLE, which exists in a selected minority of datasets (\S\ref{sec:4-15}).""")
open(P, "w").write(s)
print("OK")
