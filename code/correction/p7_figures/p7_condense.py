r"""P7 page-budget pass: condense the newly written passages (no number dropped that is not rendered
elsewhere; every removed number is listed in the run output with where it is still rendered)."""
import os, re
P = os.path.expanduser("~/papers/jams-frn50k/latex/main.tex")
s = open(P).read()


def block(a, b, new, incl_end=False):
    global s
    i = s.index(a); j = s.index(b, i) + (len(b) if incl_end else 0)
    old = s[i:j]
    gone = sorted(set(re.findall(r"\d+(?:[.,]\d+)?", old)) - set(re.findall(r"\d+(?:[.,]\d+)?", new)))
    print(f"[{a[:40]}] numbers no longer in this block: {gone}")
    s = s[:i] + new + s[j:]


block(r"\subsection{Controlled simulation: demand--capacity dependence at fixed censoring rates}", "\n\\section{Discussion}", r"""\subsection{Controlled simulation: demand--capacity dependence at fixed censoring rates}
\label{sec:controlled}

\textit{[Insert Table 12 about here.]}

\noindent Figure~\ref{fig:controlled_sim} summarizes the results.

In E2 the shared shock raises the censoring rate together with the dependence, so A4 ends in near-total censoring. A separate, fully synthetic experiment (fixed protocol, version 1.0; no FreshRetailNet-50K data) holds the expected censoring rate fixed and varies the dependence. Two latent-normal demand--capacity correlations $\eta\in\{0,0.6\}$ are crossed with three calibrated population censoring rates $q\in\{0.20,0.60,0.95\}$; each of 1,000 replicate IDs is evaluated in all six conditions (2,400 observations in four promotion--depth strata, Poisson demand, $\beta^\star=(\log 20,0.20,1.0)$). G (observed-sales Poisson) and M (censored Poisson, which ignores the dependence by design) are fitted under the validity rule of Appendix~\ref{app:repro}. The loss $A$ is the absolute error of the predicted lift $\exp(\hat\beta_P+d\hat\beta_d)-1$ in percentage points (pp), averaged over $d\in\{0.05,0.15,0.30\}$; $\Delta_{\eta q}$ is the median within-ID difference $B=A_{\text{M}}-A_{\text{G}}$, and the primary statistic $\Theta_{0.60}$ is the median within-ID change in $B$ from $\eta=0$ to $\eta=0.6$ at $q=0.60$, with 95\% Monte Carlo intervals from 2,000 joint bootstrap resamples of IDs. Design detail, failure and information diagnostics (Table A8), the oracle summary and descriptive outputs are in Online Resource 1.

All 6,000 datasets and 13,000 fits were valid ($|V_{\eta q}|=1{,}000$ in every condition) and the realized censoring matched $q$, although the median smallest uncensored stratum count fell from 313 at $q=0.20$ to 16 at $q=0.95$. M has the smaller lift error in all six conditions (Table~\ref{tab:cs_main}): $\Delta_{\eta q}$ is $-0.50$, $-4.45$ and $-17.88$ pp at $q=0.20/0.60/0.95$ with $\eta=0$, and $-0.16$, $-2.09$ and $-8.00$ pp with $\eta=0.6$. Dependence makes the comparison less favorable to M: $\Theta_{0.60}=2.56$ pp ($[2.45, 2.70]$, Monte Carlo SE 0.068, $|W_{0.60}|=1{,}000$; $\Theta_{0.20}=0.22$, $\Theta_{0.95}=10.05$). This does not make M worse than G; it reduces M's median advantage to less than half at each $q$. One result is adverse to M: at $q=0.20$ the paired depth relative error is larger for M ($+0.0030$ $[0.0011, 0.0053]$ at $\eta=0.6$; $+0.0018$ $[-0.0000, 0.0039]$ at $\eta=0$) and the promotion difference spans zero, although the lift error still favors M. The oracle's median lift loss is 1.03 pp. These results cover six specified synthetic conditions with one dependence structure; they do not establish the availability process in FreshRetailNet-50K, show that equal censoring implies equal identification, or identify causal promotion effects or optimal policies.
""")

block(r"\textbf{A4 (non-ignorable stockout): non-existence of the censored-Poisson estimate.}", r"\textbf{A5 (combined, moderate severity).}", r"""\textbf{A4 (non-ignorable stockout): non-existence of the censored-Poisson estimate.} When the capacity ceiling shares the latent demand shock, this generator censors almost every day once $\rho>0$ (median censoring share 76.7\% at $\rho=0$; 99.9\% and 99.95\% at $\rho=0.5/1.0$), and the censored-Poisson likelihood then has no finite maximum: the recession screen certifies structural non-existence in 2,080 and 2,442 of 2,700 datasets at $\rho=0.5/1.0$ (3 at $\rho=0$). Counting every attempt, M has no valid estimate in 0.8\%, 77.6\% (2,094/2,700) and 90.5\% (2,443/2,700) of datasets at $\rho=0/0.5/1.0$ (NB2 2.5\%, 86.3\%, 95.3\%; MIX 1.5\%, 77.7\%, 90.5\%; Figure~\ref{fig:a4_sensitivity}). Non-existence, not finite-sample error of a computed estimate, is the demonstrated failure mode. The earlier solver had flagged 4,352 of the 4,526 non-existence datasets as converged, which produced the large error medians and the 10 of 12 significant tests reported before this correction; on identical repeat IDs the earlier and corrected estimates agree, so the change comes from failure accounting.

On the minority of datasets with a valid M fit (6--13 simulation cells per comparison at $\rho>0$, against 23--27 at $\rho=0$), the comparison with naive and G is mixed: 5 of the 12 Holm-adjusted tests are significant, in both directions (Table~\ref{tab:e2a4}). M has larger depth error than naive and G at $\rho=0.5$ and than G at $\rho=1.0$, but smaller promotion error than naive at $\rho=1.0$ and than G at $\rho=0$. Among valid M fits at $\rho=0.5/1.0$ with both true coefficients passing the $|\beta_{\text{true}}|\geq0.05$ filter ($n=444/200$), 26.4\%/31.0\% have at least one coefficient off by more than 100\% and 2.3\%/1.5\% have both. These results condition on a finite MLE, which exists in a selected minority of datasets, and are read with the failure rates above (\S\ref{sec:4-15}).

""")

block(r"All fits are evaluated under the corrected validity rule; a fit without a valid estimate is counted as a failure, not used as an estimate.", r"generated data are identical for 28,985.",
      r"All fits are evaluated under the corrected validity rule (failures are counted, not used as estimates), on a reconstruction of the original E2 inputs (Appendix~\ref{app:repro}).", incl_end=True)

block(r"\subsection*{Reproducibility and correction note}", r"\subsection*{Hour-of-day profile}", r"""\subsection*{Reproducibility and correction note}
\label{app:repro}
Pilot criteria were fixed before the main analysis (unregistered). The E2 simulation (\S\ref{sec:3-10}) is reproduced by \texttt{main/11\_e2\_run\_all.py} (seed 42; city-cluster bootstrap $n_{\text{boot}}=4000$); the A4 fit-level error rates ($n=444/200$) by a separate script in the correction package. The real-data recovery-model sensitivity check (\S\ref{sec:3-9}) is reproduced by \texttt{main/07\_pending\_3.py} $\to$ \texttt{08\_pending\_final.py} $\to$ \texttt{10\_fix\_posthoc.py}; model-preference counts use jointly valid cells.

\textbf{Correction note.} An audit of the estimation code found numerical defects, corrected before the results in this version were computed: (i) the censored upper tail was floored at $10^{-300}$; it is now computed strictly in the log domain (also for NB2 and mixtures); (ii) the censored-row curvature had the wrong sign; (iii) convergence had been declared from solver success alone; a fit is now valid only if a recession screen finds no structural non-existence, the design is identified, and the projected gradient is $\leq10^{-6}$ on the original and standardized scales, and non-valid fits are reported, never used; (iv) the uncensored-day row mask of the hourly apportionment selected the wrong rows; (v) the store--day fixed-effect fit had no joint convergence check. The solver contract was frozen before any production fit, and 26/26 numerical checks pass. On the 249 valid empirical cells, corrected and earlier M coefficients differ by a median of $2.1\times10^{-6}$ (promotion) and $1.5\times10^{-5}$ (depth). E2 is a documented reconstruction: keys, true coefficients and anchors are identical for all 35,100 datasets and generated data for 28,985; exact bit replay of the rest was not possible because the generator calls the earlier M fit. A correction package (\texttt{repair\_log.md}, \texttt{fit\_diagnostics.csv} with every attempted fit, \texttt{before\_after.csv} with each published and corrected value) is in the replication archive.

""")

block(r"\item The originally reported profile (254 cells;", r"\end{itemize}", r"""\item The originally reported profile (top-5 hours 17, 18, 16, 19, 15; one exception at 07:00) used a wrong row mask; fixing it moves the recovered peak hour in 160 and the gap peak in 178 of 254 cells; daily totals are unchanged.

""")

old = r"\includegraphics[width=\textwidth]{fig_controlled_sim.pdf}"
assert s.count(old) == 1
s = s.replace(old, r"\includegraphics[width=0.9\textwidth]{fig_controlled_sim.pdf}")
# keep the body tables before the appendix (float flush; no spacing change)
old = "\t\\label{tab:cs_main}\n\\end{table}\n\n\\appendix"
assert s.count(old) == 1, s.count(old)
s = s.replace(old, "\t\\label{tab:cs_main}\n\\end{table}\n\n\\clearpage\n\\appendix")
open(P, "w").write(s)
print("OK")
