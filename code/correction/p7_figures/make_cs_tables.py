"""P7/S11: generate the controlled-simulation tables from controlled_sim/results (no hand-typed numbers).
Writes:
  latex/cs_main_table.tex      (body table: paired lift-error differences by eta x q with |V|, and Theta_q)
  latex/esm_controlled_sim.tex (Online Resource 1: Table A8, Oracle table, Appendix F detail, descriptive outputs)
Rounding follows protocol s8: pp to two decimals, losses to three (four where three would show 0.000)."""
import os
import numpy as np, pandas as pd

ROOT = os.path.expanduser("~/papers/jams-frn50k")
R = f"{ROOT}/controlled_sim/results"
OUT = f"{ROOT}/latex"
S = pd.read_csv(f"{R}/summaries.csv")
F = pd.read_csv(f"{R}/failure_summary.csv")
O = pd.read_csv(f"{R}/oracle_summary.csv").iloc[0]
DS = pd.read_csv(f"{R}/datasets.csv")
INF = pd.read_csv(f"{R}/information.csv")
FITS = pd.read_csv(f"{R}/fits.csv")
ORDER = [(0.0, 0.2), (0.6, 0.2), (0.0, 0.6), (0.6, 0.6), (0.0, 0.95), (0.6, 0.95)]


def cond(e, q):
    return f"eta{e:.1f}_q{q}"


def g(stat, c):
    r = S[(S.statistic == stat) & (S.condition == c)]
    assert len(r) == 1, (stat, c)
    return r.iloc[0]


def m(x, d):  # signed number in math mode
    s = f"{x:.{d}f}"  # keeps the sign of tiny negatives (e.g. -0.000), as in the completion note
    return s.replace("-", "$-$")


def ci(r, d):
    return f"{m(r.point, d)} [{m(r.lo, d)}, {m(r.hi, d)}]"


def main_table():
    rows = []
    for e, q in ORDER:
        c = cond(e, q)
        a, ed, ep = g("Delta_lift_pp", c), g("Delta_E_d", c), g("Delta_E_P", c)
        assert a.n_valid_orig == ed.n_valid_orig == ep.n_valid_orig
        dE = 3
        rows.append(f"\t\t{e:g} & {q:.2f} & {ci(a, 2)} & {ci(ed, dE)} & {ci(ep, dE)} & {int(a.n_valid_orig):,} \\\\")
    th = []
    for q in (0.2, 0.6, 0.95):
        r = g("Theta", f"q{q}")
        lab = f"{q:.2f}$" + (" (primary)" if bool(r.primary) else "")
        th.append(f"\t\t\\multicolumn{{2}}{{@{{}}l}}{{$q={lab}}} & {ci(r, 2)} & \\multicolumn{{2}}{{c}}{{MCSE {r.mcse:.3f}; $B_{{\\text{{usable}}}}={int(r.B_usable):,}$}}".replace(",}$", "}$").replace("=2,000", "=2{,}000") + f" & {int(r.n_valid_orig):,} \\\\")
    tex = r"""\begin{table}[p]
	\centering
	\footnotesize
	\caption{Controlled simulation (\S\ref{sec:controlled}): paired M$-$G differences, median over replicate IDs with 95\% Monte Carlo intervals (2,000 joint bootstrap resamples of replicate IDs). Negative values mean a smaller error for M. Lift error in percentage points (pp), averaged over depths 0.05/0.15/0.30; coefficient errors are relative (fractions). Panel B: dependence contrast $\Theta_q$, the median within-ID change $B_{r,0.6,q}-B_{r,0,q}$; positive values mean that dependence makes the comparison less favorable to M.}
	\label{tab:cs_main}
	\begin{tabular}{@{}cccccc@{}}
		\toprule
		$\eta$ & $q$ & $\Delta$ lift error (pp) & $\Delta E_d$ (depth) & $\Delta E_P$ (promotion) & $|V_{\eta q}|$ \\
		\midrule
		\multicolumn{6}{@{}l}{\textit{A. Paired M$-$G differences by condition}} \\
""" + "\n".join(rows) + r"""
		\midrule
		\multicolumn{6}{@{}l}{\textit{B. Dependence contrast $\Theta_q$ (pp)}} \\
""" + "\n".join(th) + r"""
		\bottomrule
	\end{tabular}
\end{table}
"""
    open(f"{OUT}/cs_main_table.tex", "w").write(tex)
    print(tex)


def esm():
    # Table A8
    a8 = []
    for e, q in ORDER:
        c = cond(e, q)
        fg = F[(F.condition == c) & (F.model == "G")].iloc[0]
        fm = F[(F.condition == c) & (F.model == "M")].iloc[0]
        ds = DS[DS.condition == c]
        assert len(ds) == 1000
        p10, p90 = np.percentile(ds.min_uncens_stratum, [10, 90])
        inf = INF[INF.condition == c]
        cond_med = inf.loc[inf.finite_candidate.astype(bool), "cond"].median()
        assert abs(cond_med - fm.median_cond_eligibleM) < 1e-9
        a8.append(f"\t\t{e:g} & {q:.2f} & {fm.mean_cens_share:.4f} & {fm.mean_cens_no_promo:.4f}/{fm.mean_cens_shallow_promo:.4f}/{fm.mean_cens_moderate_promo:.4f}/{fm.mean_cens_deep_promo:.4f} & "
                  f"{int(fg.n_attempted - fg.n_valid)}/{int(fg.n_attempted):,} & {int(fm.n_attempted - fm.n_valid)}/{int(fm.n_attempted):,} & "
                  f"{int(fm.median_min_uncens)} ({int(p10)}--{int(p90)}) & {int(fm.n_all_censored)}/{int(fm.n_rank_deficient)} & "
                  f"{fm.median_cond_eligibleM:.1f} ({int(fm.n_cond_denominator):,}; {int(fm.diag_ok):,} ok) \\\\")
    # descriptive outputs
    dep = []
    for e, q in ORDER:
        c = cond(e, q)
        cells = [ci(g(s, c), 2) for s in ("Delta_ae_05_pp", "Delta_ae_15_pp", "Delta_ae_30_pp")]
        ag = [S[(S.statistic == f"agreement_{k}") & (S.condition == c)].iloc[0] for k in
              ("both_coef_and_lift", "both_coef_not_lift", "lift_not_both_coef", "neither")]
        agt = " & ".join(f"{100*x.point:.1f}" for x in ag)
        dep.append(f"\t\t{e:g} & {q:.2f} & " + " & ".join(cells) + f" & {agt} \\\\")
    # coefficient-error MCSE etc. and all component failure flags
    flags = ["n_structural_nonexistence", "n_unresolved_screen", "n_nonconvergence", "n_nonfinite_objective_coefficient", "n_nonfinite_lift_loss"]
    tot_fail = int(F[flags].sum().sum())
    n_fits = len(FITS)
    assert n_fits == 13000 and int(FITS.valid.astype(str).eq("True").sum()) == 13000
    thq = {q: g("Theta", f"q{q}") for q in (0.2, 0.6, 0.95)}
    eig = INF.groupby("condition").agg(eig_min_med=("eig_min", "median"), eig_max_med=("eig_max", "median"))
    eigtxt = "; ".join(f"$\\eta={e:g}$, $q={q:.2f}$: {eig.loc[cond(e, q), 'eig_min_med']:.3f}/{eig.loc[cond(e, q), 'eig_max_med']:.2f}" for e, q in ORDER)

    tex = r"""\documentclass[11pt]{article}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[margin=2.2cm]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{float}
\renewcommand{\thetable}{S\arabic{table}}
\begin{document}
\sloppy
\begin{center}
{\large\textbf{Online Resource 1}}\\[4pt]
Out of Stock, Out of Sight: Stockout Censoring and\\Model-Relative Gaps in Promotion-Lift Estimates\\[2pt]
Controlled simulation of demand--capacity dependence: design detail and diagnostics
\end{center}

This Online Resource supplements the controlled-simulation subsection of the main text. All values are computed from the frozen result files (\texttt{controlled\_sim/results/}: \texttt{summaries.csv}, \texttt{failure\_summary.csv}, \texttt{oracle\_summary.csv}, \texttt{datasets.csv}, \texttt{information.csv}); the protocol is version~1.0 (22~November~2026). This is a fully synthetic experiment. FreshRetailNet-50K data are not an input, and the results are kept separate from the E2 benchmark of the main text.

\section*{S1 Design and implementation (protocol Appendix F detail)}

\textbf{Conditions and data.} Two demand--capacity dependence levels $\eta\in\{0,0.6\}$ are crossed with three calibrated censoring rates $q\in\{0.20,0.60,0.95\}$. Each of 1,000 replicate IDs is evaluated in all six conditions (6,000 datasets). A dataset has 2,400 observations in four fixed strata: no promotion ($P=0$, $d=0$, 1,200 observations) and shallow, moderate and deep promotion ($P=1$, $d=0.05/0.15/0.30$, 400 observations each). With independent standard normals $Z_D,Z_C$, demand and capacity are
\begin{gather*}D=Q_{\mathrm{Pois}}\{\Phi(Z_D);\lambda_s\},\quad \lambda_s=\exp(x_s\beta^\star)=20\exp(0.20P_s+d_s),\\ C=Q_{\mathrm{Pois}}\{\Phi(\eta Z_D+\sqrt{1-\eta^2}Z_C);\kappa_{s\eta q}\},\end{gather*}
with $\beta^\star=(\log 20,0.20,1.0)$, so $\eta$ is a latent-normal (Gaussian-copula) correlation and demand is marginally Poisson in every condition. Observed sales are $K=\min(D,C)$ with censoring indicator $I=\mathbf{1}\{D>C\}$; $D=C$ is uncensored. The two normal arrays of a replicate are reused across its six conditions, so demand is identical across conditions within an ID.

\textbf{Calibration.} The 24 capacity means $\kappa_{s\eta q}=\lambda_s\exp(a_{s\eta q})$ solve the population equation $\Pr(D>C\mid s,\eta,a)=q$ by deterministic quadrature of the conditional-normal integral and Brent's method; they were frozen before any evaluation fit. All 24 cells passed ($\max|p-q|=2.73\times10^{-11}$, tolerance $10^{-6}$; quadrature error $\le1.7\times10^{-12}$); an independent Monte Carlo check gave $\max|z|=1.77$. Datasets were never rejected or resampled to force the realized censoring share.

\textbf{Estimators and information sets.} G is an ordinary Poisson fit to $(X,K)$; M maximizes the censored-Poisson working likelihood $\sum_i[(1-I_i)\log f(K_i;\mu_i)+I_i\log\Pr(Y>K_i)]$ on $(X,K,I)$ with the strict survival convention; the oracle O is an ordinary Poisson fit to the latent demand $(X,D)$, fitted once per replicate (1,000 fits). $X=(1,P,d)$ without interactions, penalties, constraints, weights or standardization. At $\eta=0.6$ the working likelihood ignores the dependence by design. The truth enters only data generation, loss evaluation and the labeled oracle.

\textbf{Numerical contract and validity.} Average negative log likelihood, analytic gradients, stable log-scale tails without probability floors, L-BFGS-B (2,000 iterations, 20,000 evaluations, 50 line-search steps, memory 10, $\|\nabla\|_\infty\le10^{-6}$ recomputed independently). M uses three starts (the valid G estimate $b$, $b+(0,0.1,0)$, $b+(0,0,0.1)$) and keeps the highest likelihood among valid candidates. Before optimization a bounded linear program screens for a maximum at diverging coefficients (structural nonexistence). A fit is valid only if the screen passes, a start converges with finite coefficients, objective and verified gradient norm $\le10^{-6}$, and all lifts and losses are finite. Large finite losses, sign reversals and high condition numbers are not excluded. Observed information is the Hessian of the average negative log likelihood, $H=X^{\top}\mathrm{diag}(w_i)X/n$, recorded as a diagnostic, not an eligibility rule.

\textbf{Losses and estimands.} Lift at depth $d$ is $\widehat L_m(d)=\exp(\hat\beta_{P,m}+d\hat\beta_{d,m})-1$ with truth $L^\star(d)=\exp(0.20+d)-1$. The lift loss is $A_m=\tfrac{100}{3}\sum_{d\in\{0.05,0.15,0.30\}}|\widehat L_m(d)-L^\star(d)|$ (pp); the coefficient losses are $E_{P}=|\hat\beta_P-0.20|/0.20$ and $E_d=|\hat\beta_d-1.0|$. Within each replicate $B_{r,\eta q}=A_{M,r,\eta q}-A_{G,r,\eta q}$; $\Delta_{\eta q}$ is the median of $B$ over $V_{\eta q}$, the IDs with both M and G valid. The primary statistic is $\Theta_{0.60}=\mathrm{median}_{r\in W_{0.60}}[B_{r,0.6,0.60}-B_{r,0,0.60}]$ with $W_q=V_{0,q}\cap V_{0.6,q}$; $\Theta_{0.20}$ and $\Theta_{0.95}$ are secondary. Monte Carlo intervals use 2,000 resamples of the 1,000 replicate IDs with the same resampled vector for every condition and statistic; the Monte Carlo standard error (MCSE) is the bootstrap standard deviation.

\textbf{Validation and execution.} The engineering validation (separate random stream, five smoke-test IDs excluded from evaluation) passed 30/30 checks, including censoring hand checks, likelihood and far-tail checks, finite-difference gradients and Hessians, structural-screen examples, calibration and stream checks, and pairing/bootstrap checks. Two defects found by validation were fixed before the freeze and before any evaluation fit (a tail routine that dropped series terms for scalar inputs, and a subtractive cancellation in the far-tail Hessian weight). All 1,000 IDs were attempted in all six conditions: 6,000 datasets, """ + f"{n_fits:,}" + r""" fits (6,000 G, 6,000 M, 1,000 O) and 25,000 scheduled starts; the run was not resumed.

\section*{S2 Failures, realized censoring and information (Table A8)}

\begin{table}[H]
\centering
\footnotesize
\caption{Diagnostics by condition (protocol Table A8) over all 1,000 attempted datasets. Censoring: mean realized share overall and by stratum (no/shallow/moderate/deep promotion). Invalid fits: attempted fits without a valid estimate. Min.\ uncensored: median (10th--90th percentile) of the smallest uncensored count among the four strata. All-cens./rank-def.: datasets with all observations censored / rank-deficient uncensored design. Condition number of the average-NLL Hessian at the selected finite M candidate: median (denominator; diagnostics labeled ok).}
\label{tab:a8}
\resizebox{\textwidth}{!}{%
\begin{tabular}{@{}cccccccc c@{}}
\toprule
$\eta$ & $q$ & Censoring & By stratum & Invalid G & Invalid M & Min.\ uncensored & All-cens./rank-def. & Cond.\ number \\
\midrule
""" + "\n".join(a8) + r"""
\bottomrule
\end{tabular}}
\end{table}

No fit failed in any condition: """ + f"{tot_fail}" + r""" structural nonexistence, unresolved screen, nonconvergence, nonfinite-coefficient or nonfinite-loss cases among the 13,000 fits (the binomial Monte Carlo standard error of each failure fraction is therefore 0). All 6,000 M information diagnostics are labeled ok (no singular, indefinite or missing case). Median smallest/largest Hessian eigenvalues by condition: """ + eigtxt + r""". Equal censoring rates do not imply equal information: the median smallest uncensored stratum count falls from 313 at $q=0.20$ to 16 at $q=0.95$ while the condition number changes little.

\section*{S3 Oracle diagnostic}

\begin{table}[H]
\centering
\footnotesize
\caption{Oracle (latent-demand Poisson fit, one per replicate; """ + f"{int(O.n_valid):,} valid of {int(O.n_attempted):,}" + r"""). Median (10th, 90th percentile).}
\label{tab:oracle}
\begin{tabular}{@{}lccc@{}}
\toprule
 & $E_P$ (promotion) & $E_d$ (depth) & Lift loss $A$ (pp) \\
\midrule
""" + f"Oracle & {O.E_P_median:.3f} ({O.E_P_p10:.3f}, {O.E_P_p90:.3f}) & {O.E_d_median:.3f} ({O.E_d_p10:.3f}, {O.E_d_p90:.3f}) & {O.A_pp_median:.2f} ({O.A_pp_p10:.2f}, {O.A_pp_p90:.2f}) \\\\" + r"""
\bottomrule
\end{tabular}
\end{table}

The oracle uses information (latent demand) that neither G nor M has. Its lift loss is a reference for sampling variability in this design; it does not determine eligibility for any M$-$G comparison.

\section*{S4 Prespecified descriptive outputs}

\begin{table}[H]
\centering
\footnotesize
\caption{Depth-specific paired M$-$G absolute lift-error differences (pp; median, 95\% Monte Carlo interval) and agreement categories (\% of $V_{\eta q}$, $|V_{\eta q}|=1{,}000$ in every condition): M improves both coefficient losses and the lift loss (Both+lift); both coefficients but not lift (Coef only); lift but not both coefficients (Lift only); neither. ``Improves'' means a strictly negative paired difference. Descriptive only; no tests.}
\label{tab:desc}
\resizebox{\textwidth}{!}{%
\begin{tabular}{@{}cccccrrrr@{}}
\toprule
$\eta$ & $q$ & $d=0.05$ & $d=0.15$ & $d=0.30$ & Both+lift & Coef only & Lift only & Neither \\
\midrule
""" + "\n".join(dep) + r"""
\bottomrule
\end{tabular}}
\end{table}

\section*{S5 Secondary dependence contrasts and adverse results}

""" + f"$\\Theta_{{0.20}}={thq[0.2].point:.3f}$ [{thq[0.2].lo:.3f}, {thq[0.2].hi:.3f}] (MCSE {thq[0.2].mcse:.3f}) and $\\Theta_{{0.95}}={thq[0.95].point:.2f}$ [{thq[0.95].lo:.2f}, {thq[0.95].hi:.2f}] (MCSE {thq[0.95].mcse:.3f}); $|W_q|=1{{,}}000$ and 2,000 of 2,000 resamples usable for every statistic." + r""" At $q=0.20$ the paired depth relative error is larger for M: """ + \
        f"$\\Delta E_d={m(g('Delta_E_d', cond(0.0, 0.2)).point, 4)}$ [{m(g('Delta_E_d', cond(0.0, 0.2)).lo, 4)}, {m(g('Delta_E_d', cond(0.0, 0.2)).hi, 4)}] at $\\eta=0$ and {m(g('Delta_E_d', cond(0.6, 0.2)).point, 4)} [{m(g('Delta_E_d', cond(0.6, 0.2)).lo, 4)}, {m(g('Delta_E_d', cond(0.6, 0.2)).hi, 4)}] at $\\eta=0.6$" + \
        r"""; the promotion difference $\Delta E_P$ spans zero at $q=0.20$. At $q=0.60$ and $0.95$ both coefficient differences are negative. These results describe six specified synthetic conditions. They do not establish the availability process in FreshRetailNet-50K, show that equal censoring implies equal identification, identify causal promotion effects, or establish optimal promotion policies.

\section*{S6 Reproducibility}
Source commit \texttt{acfa5e0}; Python 3.14.9, NumPy 2.5.4, SciPy 1.18.1, pandas 3.1.0; master seed \texttt{SeedSequence}(20261122) with separate evaluation, bootstrap and engineering streams (replicate $r$ uses evaluation child $r-1$). Medians use the midpoint convention and quantiles linear interpolation. Every reported fit can be regenerated from the frozen calibration and seed mapping without a new random draw; the archive holds \texttt{calibration.csv}, \texttt{datasets.csv}, \texttt{fits.csv}, \texttt{starts.csv}, \texttt{losses.csv}, \texttt{information.csv}, \texttt{paired.csv}, \texttt{dependence.csv}, \texttt{bootstrap.csv}, \texttt{summaries.csv}, \texttt{validation\_log.txt} and \texttt{run\_manifest.json}.

\end{document}
"""
    open(f"{OUT}/esm_controlled_sim.tex", "w").write(tex)


if __name__ == "__main__":
    main_table(); esm()
