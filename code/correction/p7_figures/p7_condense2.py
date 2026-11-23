r"""P7 page-budget pass 2: tighten Sec. 5 (4-15), Sec. 3.9 and the appendix alternative-model text."""
import os, re
P = os.path.expanduser("~/papers/jams-frn50k/latex/main.tex")
s = open(P).read()


def block(a, b, new, incl_end=False):
    global s
    assert s.count(a) == 1, a[:50]
    i = s.index(a); j = s.index(b, i) + (len(b) if incl_end else 0)
    old = s[i:j]
    gone = sorted(set(re.findall(r"\d+(?:[.,]\d+)?", old)) - set(re.findall(r"\d+(?:[.,]\d+)?", new)))
    print(f"[{a[:40]}] numbers no longer in this block: {gone}")
    s = s[:i] + new + s[j:]


block(r"\S\ref{sec:3-10}'s simulation benchmark found M's distributional robustness", r"\noindent\textbf{The decomposition procedure", r"""\S\ref{sec:3-10}'s benchmark found M robust to misspecified demand distributions and censoring-label noise (A0--A3). A4 differs: when the capacity ceiling is driven by the same latent shock as demand, the tested generator censors nearly every day once $\rho>0$, and the censored-Poisson likelihood then typically has no finite maximum (structural non-existence in 2,080 and 2,442 of 2,700 datasets at $\rho=0.5/1.0$). This non-existence, not a finite-sample error of a computed estimate, is the demonstrated failure mode; NB2 and MIX fail at equal or higher rates. Where a finite MLE exists, the M-versus-naive/G comparison is mixed (Table~\ref{tab:e2a4}).

\textbf{Scope, stated precisely.} This is a failure under one generative construction (a shared latent shock linking demand and capacity), not a general law that ``all non-ignorable censoring breaks M.'' The per-dataset censoring shares resolve the earlier open question whether the extreme A4 errors reflected bias or near-total censoring: they reflect near-total censoring and non-existence. Not established are (1) whether real FreshRetailNet-50K stockouts share this property (we have no instrument for latent demand; the real panels, with a median stockout-day share of 58.8\% and a valid M fit in 249 of 254 cells, are far from near-total censoring), and (2) how non-ignorability affects M at moderate censoring, which the controlled simulation (\S\ref{sec:controlled}) addresses for one dependence structure only, finding a smaller advantage for M rather than a reversal.

\textbf{Resulting recommendation.} Because no validated pre-test for non-ignorable censoring is presented here, we do not prescribe withholding M. Instead, report the naive/G and M-corrected estimates side by side (dual reporting) whenever ignorability has not been independently verified; confirm that M has a finite, valid maximum before reading it, and report a non-existent or non-valid fit as such rather than substituting a number; and treat divergence between the two estimates as the diagnostic signal. The absence of a specific reason to suspect non-ignorable stockout timing (e.g., promotions timed to expected demand, supply shortages correlated with anticipated spikes) does not verify the assumption. This is narrower than \S\ref{sec:3-8}'s original unconditional-substitution claim: robustness to the misspecifications tested in A0--A3 stands, but it does not extend to non-ignorable censoring (A4), where ignorability is an unverified assumption that dual reporting is designed to surface, not assume away.

""")

block(r"On uncensored days, Pearson $\chi^2/\mathrm{df}$ has median 3.22 (249 cells).", r"Table~\ref{tab:auditstatus} consolidates audit and sensitivity status;", r"""On uncensored days, Pearson $\chi^2/\mathrm{df}$ has median 3.22 (249 cells). Under the corrected solver contract the censored recovery alternatives are valid in most cells: NB2 252/254, ZIP 238/254 (intercept-only inflation) and 237/254 (promotion-dependent), mixture censoring 245/247/248 at fixed $\pi=0.95/0.85/0.75$ and 248 with estimated $\pi$; the earlier counts (NB2 16, ZIP 9, mixtures 20--168) reflected the defective solver (Appendix~\ref{app:repro}). With each alternative as the recovery reference, the observed-sales-versus-recovery gaps satisfy the pre-fixed lower-bound rule for both coefficients (Appendix~\ref{app:recovery}). These are recovery-reference sensitivity checks, not matched alternative-family comparisons; the headline gap remains conditional on the Poisson recovery model. """)

block(r"\textbf{Gaps under the alternatives (valid fits).}", r"The specifications were fixed before the fits:", r"""\textbf{Gaps under the alternatives (valid fits).} Median $|b_{\text{G}}-b_{\text{alt}}|/|b_{\text{alt}}|$ with Rademacher wild-cluster 95\% intervals (18 cities) and share $\geq$20\%, on the cells where each alternative is valid: NB2 depth 0.445 [0.361, 0.530], 0.754, promotion 0.615 [0.520, 0.704], 0.829; ZIP (intercept-only) depth 0.288 [0.245, 0.330], 0.656, promotion 0.598 [0.520, 0.677], 0.794; fixed-$\pi$ mixtures depth 0.253--0.265 (lower bounds $\geq$0.224, shares $\geq$0.589), promotion 0.501--0.510 (lower bounds $\geq$0.394, shares $\geq$0.761); estimated $\hat\pi$ depth 0.260 [0.230, 0.290], promotion 0.524 [0.408, 0.639]. The NB2-reference depth gap is larger partly because the reference distribution also changes. Earlier converged-subset results (16 NB2 and 9 ZIP cells) came from non-valid fits and are superseded. Every gap remains model-relative (\S\ref{sec:2-4}). """)

block(r"\textbf{Decision rule and what it shows.}", "\\clearpage\n\\bibliography", r"""\textbf{Decision rule and what it shows.} The 25.9\%/52.5\% medians \emph{survive} an alternative if, on the cells where it is valid, the median gap is at least 20\%, the share above 20\% is at least 50\%, and the cluster-interval lower bound exceeds 20\%. Every alternative satisfies the rule for both coefficients, which revises the earlier statement that NB2/ZIP convergence was too sparse to apply it. This shows that the observed-versus-recovered gap is not specific to the Poisson recovery family, not that any recovery is closer to true demand, and it is not a matched alternative-family comparison (observed-sales NB2 fits are not recorded). We report both estimates conditional on the recovery model (\S\ref{sec:4-13}).

\textbf{Model preference (run).} On jointly valid cells, M beats the mixture-censoring specifications by AIC/BIC/LRT in every cell (242/245/244 at $\pi=0.95/0.85/0.75$, 244 with $\hat\pi$; median $\Delta$AIC $+226$/$+718$/$+1{,}277$/$+3.4$). Zero inflation is small ($\hat{\psi}$ median 0.004), though ZIP is AIC-preferred in 231 of 237 cells (median $\Delta$AIC $-128$). NB2 is the data-preferred family (249/249 cells, median $\Delta$AIC $-4{,}497$; NB2+mixture 245/247, 221/247, 192/248 and 249/249 at $\pi=0.95/0.85/0.75$ and $\hat\pi$), a specification diagnostic, not a claim about the size of the Poisson-M gap.

""")
open(P, "w").write(s)
print("OK")
