r"""P7 + S11: apply the correction edits to latex/main.tex.
Every replacement asserts the exact number of matches; aborts without writing on any mismatch.
Numbers come from p7_*.out / before_after.csv / controlled_sim/results (see the .out files)."""
import os, re, sys
from collections import Counter

P = os.path.expanduser("~/papers/jams-frn50k/latex/main.tex")
src = open(P).read()
orig = src
E = []  # (old, new, expected_count)
BLK = []  # (start, end, new): whole-block rewrites, applied after token replacements


def R(old, new, n=1):
    E.append((old, new, n))


# ---------------------------------------------------------------- token-level (gap numbers)
R(r"25.3\%", r"25.9\%", 11)
R(r"52.4\%", r"52.5\%", 10)
R(r"59.8\%", r"60.2\%", 6)
R(r"77.2\%", r"77.1\%", 6)
R(r"50.2\%", r"50.0\%", 7)
R(r"84.1\%", r"83.4\%", 6)
R(r"38.4\%", r"38.3\%", 2)
R(r"30.5\%", r"30.6\%", 2)
R(r"(median 38.3\%/52.1\%)", r"(median 38.3\%/52.5\%)")
R(r"(median 30.6\%/31.0\% on the common denominator)", r"(median 30.6\%/30.5\% on the common denominator)")
R("0.253", "0.259", 5)
R("0.524", "0.525", 4)
R("0.393", "0.378", 3)
R("0.588", "0.587", 3)
R(r"covariate-omission component $0.311/0.514$", r"covariate-omission component $0.304/0.511$")
R(r"80.3\%", r"79.9\%", 3)
R(r"45.3\%", r"45.0\%", 2)
R("1.494", "1.512", 2)
R(r"median gaps of 25--52\% across elasticity coefficients", r"median gaps of 26--52\% across elasticity coefficients")

# ---------------------------------------------------------------- abstract
R(r"the matched censored-versus-observed Poisson contrast across 254 city-category panels has median gaps",
  r"the matched censored-versus-observed Poisson contrast across the 249 of 254 city-category panels with a valid censored fit has median gaps")
R(r"In the tested nonzero shared-shock settings, Poisson recovery has higher coefficient-estimation error than observed-sales estimators. The empirical gaps are associational and relative to a Poisson recovery model, not errors against latent demand; some alternatives have convergence limitations.",
  r"In a misspecified-generator benchmark, strongly non-ignorable stockouts censor almost every day and the censored-Poisson estimate then usually does not exist; in a controlled simulation at fixed censoring rates, demand--capacity dependence narrows but does not reverse its lift-error advantage over an observed-sales estimator. The empirical gaps are associational and relative to a Poisson recovery model, not errors against latent demand.")

# ---------------------------------------------------------------- contributions item 2
R(r"""At $\rho\in\{0.5,1.0\}$ in A4, M, NB2, and MIX show higher descriptive median relative error than naive/G; the eight Holm-adjusted tests cover the M-versus-naive/G comparisons only, with NB2/MIX differences left descriptive. The result describes the tested configurations, not a general correction-failure boundary, and does not establish this mechanism for real FreshRetailNet-50K stockouts (\S\ref{sec:4-15}).""",
  r"""Under A4 with $\rho\in\{0.5,1.0\}$ the generator censors almost every day and the censored-Poisson MLE does not exist in most datasets (M has no valid estimate in 77.6\%/90.5\% of attempts); on the minority with a finite MLE the comparison with naive/G is mixed (5 of 12 Holm-adjusted tests significant, in both directions). A separate controlled simulation (\S\ref{sec:controlled}) holds the expected censoring rate fixed and varies demand--capacity dependence: M keeps the smaller lift error in all six conditions, and dependence reduces that advantage ($\Theta_{0.60}=2.56$ pp). Both results describe the tested configurations, not a general correction-failure boundary, and do not establish these mechanisms for real FreshRetailNet-50K stockouts (\S\ref{sec:4-15}).""")

# ---------------------------------------------------------------- 2.2 / 2.4 / 2.5 / 2.7
R(r"Mixture fits yield similar gaps where they converge, with convergence limitations; the primary estimates remain conditional",
  r"Mixture fits are valid in 245--248 of 254 cells and yield similar gaps; the primary estimates remain conditional")
R(r"""MLE by L-BFGS-B, initialized with 10 IRLS iterations of a Poisson GLM on uncensored days (a numerical initialization only; selection into uncensored days can depend on demand, and consistency of this GLM is not established). All 254 cells converge (median 144 iterations).""",
  r"""MLE by L-BFGS-B on the average negative log likelihood with a strictly computed upper tail (no probability floor), started from 10 IRLS iterations of a Poisson GLM on uncensored days and two perturbed starts (a numerical initialization only; selection into uncensored days can depend on demand, and consistency of this GLM is not established). A fit is valid only if a recession screen finds no direction of unbounded likelihood, the design is identified, and the projected gradient is at most $10^{-6}$ on both the original and the column-standardized scale; 249 of the 254 cells are valid (median 196 iterations; Appendix~\ref{app:repro}).""")
R(r"""\item \textbf{F (sensitivity).} M with store--day fixed effects $\alpha$: alternating MLE: (b | $\alpha$) by the censored Poisson MLE with offset $\alpha$, ($\alpha$ | b) by Gauss--Newton with step-halving to guarantee monotone likelihood. Run on the 20 cells with the largest promotion $\times$ stockout overlap.""",
  r"""\item \textbf{F (sensitivity).} M with store--day fixed effects $\alpha$, estimated jointly by damped Newton steps on the slopes and offsets with backtracking and a joint stationarity check (slope and per-group offset scores $\leq10^{-6}$). Within a store--day only promo, depth and activity vary; the other covariates are absorbed by $\alpha$. Store--days on which every row is censored have $\hat\alpha=+\infty$ and carry no recoverable demand; they are counted, not fitted. Run on the 20 cells with the largest promotion $\times$ stockout overlap (the originally reported alternating fit is superseded; Appendix).""")
R(r"$\mathrm{VIF}_{\mathrm{depth}}>10$ in 6.3\% of cells; the promo--depth collinearity is structural (depth $=$ promo $\times$ (1 $-$ discount), zero off promo days) and persists under centering (max slope shift $5\times10^{-4}$)",
  r"$\mathrm{VIF}_{\mathrm{depth}}>10$ in 6.0\% of the 249 cells; the promo--depth collinearity is structural (depth $=$ promo $\times$ (1 $-$ discount), zero off promo days) and persists under centering (max slope shift $4\times10^{-6}$)")
R(r"convergence (M: 167 iterations; F: 26,010 store--day groups)",
  r"convergence (M: valid, 181 iterations under the corrected solver; F: jointly valid, 26,010 store--day groups)")
R(r"the recovery-model and fixed-effect fits with convergence limitations, and the completed wild-cluster analysis",
  r"the recovery-model and fixed-effect sensitivity fits, the completed wild-cluster analysis, and the controlled simulation (\S\ref{sec:controlled})")

# ---------------------------------------------------------------- Section 3 header
R(r"""All estimates: 254 cells (city $\times$ first-level category) with sufficient data; 90 days each (2024-03-28 to 06-25); seed 42, deterministic. M-model converged in all 254 cells (median 144 L-BFGS iterations). Sales in normalized units (1 = 0.1 unit).""",
  r"""All estimates: 254 cells (city $\times$ first-level category) with sufficient data; 90 days each (2024-03-28 to 06-25); seed 42, deterministic. Under the corrected validity rule (\S\ref{sec:2-4}) the M model is valid in 249 of the 254 cells (median 196 L-BFGS iterations): cells 0$\times$8, 10$\times$4 and 17$\times$21 do not converge, 4$\times$28 is unidentified, and 14$\times$31 has no finite maximum. Every M-based statistic below is computed on these 249 cells. Sales in normalized units (1 = 0.1 unit).""")

# ---------------------------------------------------------------- 3.2 direction
R(r"""184/254 cells (72.4\%) show \emph{attenuation}: the naive estimate is closer to zero than the recovered effect (170 cells with a positive naive effect understated; 14 cells with a negative naive effect understated, the pilot cell among them). 55 cells (21.7\%) show exaggeration in magnitude, and 15 cells (5.9\%) flip sign (13 naive-positive $\rightarrow$ recovered-negative; 2 the reverse). Only 9.1\% of cells show""",
  r"""181/249 cells (72.7\%) show \emph{attenuation}: the naive estimate is closer to zero than the recovered effect (167 cells with a positive naive effect understated; 14 cells with a negative naive effect understated, the pilot cell among them). 53 cells (21.3\%) show exaggeration in magnitude, and 15 cells (6.0\%) flip sign (13 naive-positive $\rightarrow$ recovered-negative; 2 the reverse). Only 9.1\% of the 254 cells show""")
R(r"differs from the recovered estimate by 46\% of its magnitude (a model-relative gap). (This 15/254 = 5.9\% is the P1a",
  r"differs from the recovered estimate by 45\% of its magnitude (a model-relative gap). (This 15/249 = 6.0\% is the P1a")

# ---------------------------------------------------------------- 3.4
R(r"An exploratory, post-hoc model-relative decomposition (same 254 cells)", r"An exploratory, post-hoc model-relative decomposition (the 249 cells with a valid corrected M)")
R(r"In $3$ depth / $42$ promo cells $|b_{\text{M}}|<0.05$", r"In $2$ depth / $42$ promo cells $|b_{\text{M}}|<0.05$")
R(r"the censoring swap evaluated at the reduced covariate set (reduced-spec recovery fits, $254/254$ converged), was refit",
  r"the censoring swap evaluated at the reduced covariate set (reduced-spec recovery fits valid in $251/254$ cells for depth and $248/254$ for promotion; $248$ and $245$ cells with both fits valid), was refit")
R(r"The promotion coefficient is negative in 10.6\% of cells under naive OLS but in 29.9\% under the recovered model",
  r"The promotion coefficient is negative in 10.8\% of cells under naive OLS but in 30.1\% under the recovered model")
R(r"is 28/254 = 11.0\% for depth and 34/254 = 13.4\% for promo", r"is 27/249 = 10.8\% for depth and 34/249 = 13.7\% for promo")
R(r"(naive-reduced vs. M: 59/254 = 23.2\% depth, 57/254 = 22.4\% promo;", r"(naive-reduced vs. M: 56/249 = 22.5\% depth, 56/249 = 22.5\% promo;")
R(r"(15/254 = 5.9\%, naive vs. recovered mean promotion effect", r"(15/249 = 6.0\%, naive vs. recovered mean promotion effect")
R(r"(i) P1a mean-effect flip 5.9\%, (ii) matched Poisson depth-coefficient flip 11.0\%, (iii) matched Poisson promo-coefficient flip 13.4\%.",
  r"(i) P1a mean-effect flip 6.0\%, (ii) matched Poisson depth-coefficient flip 10.8\%, (iii) matched Poisson promo-coefficient flip 13.7\%.")
R(r"P1a $|bias|/|naive|$) = -0.022, n=254)", r"P1a $|bias|/|naive|$) = -0.023, n=249)")
R(r"also changes \emph{which} of the 254 panels", r"also changes \emph{which} of the 249 panels")
R(r"For the depth coefficient, 90/254 cells (35.4\%) change flag status (66 flagged only under the unmatched sequencing, 24 only under the matched sequencing); for the promotion coefficient, 64/254 cells (25.2\%) change status",
  r"For the depth coefficient, 86/249 cells (34.5\%) change flag status (63 flagged only under the unmatched sequencing, 23 only under the matched sequencing); for the promotion coefficient, 64/249 cells (25.7\%) change status")
R(r"(number remaining noted below; 254/254 at $d=3.6\%$ and 8.7\%, 253/254 at 14.4\%, 239/254 at 23.4\%, 172/254 at 33.4\%).",
  r"(computed on the 249 cells with a valid corrected M; number remaining noted below: 249/249 at $d=3.6\%$, 8.7\% and 14.4\%, 236/249 at 23.4\%, 169/249 at 33.4\%).")
R(r"""		3.6\%  & 7.00                   & [4.42, 8.95]      & 254            \\
		8.7\%  & 4.19                   & [1.72, 7.53]      & 254            \\
		14.4\% & 3.09                   & [1.03, 5.52]      & 253            \\
		23.4\% & 1.90                   & [$-$2.13, 4.48]   & 239            \\
		33.4\% & 0.47                   & [$-$5.04, 8.04]   & 172            \\""",
  r"""		3.6\%  & 6.98                   & [4.42, 8.88]      & 249            \\
		8.7\%  & 3.87                   & [1.51, 7.47]      & 249            \\
		14.4\% & 2.96                   & [1.03, 4.96]      & 249            \\
		23.4\% & 1.66                   & [$-$2.20, 4.31]   & 236            \\
		33.4\% & 0.16                   & [$-$5.04, 8.04]   & 169            \\""")
R(r"modest in absolute (percentage-point) terms (0.5--7.0 percentage points)", r"modest in absolute (percentage-point) terms (0.2--7.0 percentage points)")

# ---------------------------------------------------------------- 3.3 hourly, 3.5 dose, 3.6 F
R(r"""The observed-sales estimate is below the recovered estimate at 23 of 24 hours, with the largest gap in the evening; the single exception is 07:00, where the naive estimate is marginally above the recovered one (naive 0.2953 vs. recovered 0.2853). The hourly profile is reported in Appendix Table~\ref{tab:hourly}.""",
  r"""With the corrected uncensored-day row mask and the corrected M (249 cells), the observed-sales estimate is below the recovered estimate at 22 of 24 hours, with the largest gap at 20:00; the exceptions are 07:00 and 08:00, where the naive estimate is marginally above the recovered one (0.2784 vs.\ 0.2701 and 0.4753 vs.\ 0.4696). The originally reported profile used a row mask that selected the wrong days for the uncensored-day hour profile; fixing it changes the peak hour of recovered demand in 160 of 254 cells and the peak hour of the gap in 178 of 254, while daily totals are unchanged. The hourly profile is reported in Appendix Table~\ref{tab:hourly}.""")
R(r"Pooled promotion store-days across all 254 cells ($\approx$2.29M rows)", r"Pooled promotion store-days across the 249 cells with a valid corrected M ($\approx$2.18M rows)")
R(r"four quintiles lie between -0.442 and -0.480, with a smaller magnitude in the lowest-depth quintile (-0.349).",
  r"four quintiles lie between -0.447 and -0.481, with a smaller magnitude in the lowest-depth quintile (-0.353).")
R(r"jointly converged cells. The existing store--day and product-effect checks are not this comparison and do not establish its robustness. The appendix reports their scope and convergence limitations.",
  r"jointly converged cells. The existing store--day and product-effect checks are not this comparison and do not establish its robustness. The corrected store--day fit is jointly valid in all 20 cells (Table~\ref{tab:fmodel}); the appendix reports its scope.")

# ---------------------------------------------------------------- 3.8
R(r"(city-block bootstrap 95\% CI $[76.0\%, 84.9\%]$, n=18 cities)", r"(city-block bootstrap 95\% CI $[75.5\%, 84.6\%]$, n=18 cities)")
R(r"60.2\% $[55.7\%, 64.4\%]$ / 77.1\% $[72.5\%, 81.8\%]$", r"60.2\% $[56.1\%, 64.8\%]$ / 77.1\% $[72.4\%, 81.6\%]$")
R(r"differs from the recovered estimate by more than 45\% of its magnitude (P1a median model-relative gap ratio 45.6\%, CI $[41.9\%, 51.9\%]$)",
  r"differs from the recovered estimate by about 45\% of its magnitude (P1a median model-relative gap ratio 45.5\%, CI $[40.9\%, 51.8\%]$)")
R(r"11.0\% $[7.5\%, 15.0\%]$ (depth) / 13.4\% $[10.4\%, 16.1\%]$ (promo)", r"10.8\% $[7.0\%, 15.3\%]$ (depth) / 13.7\% $[10.6\%, 16.4\%]$ (promo)")
R(r"23.2\% $[18.4\%, 28.0\%]$ / 22.4\% $[17.4\%, 28.1\%]$", r"22.5\% $[17.7\%, 27.4\%]$ / 22.5\% $[17.5\%, 28.1\%]$")
R(r"intervals are $[0.226,0.282]$ for depth and $[0.405,0.642]$ for promotion; exceedance-share lower bounds are 0.556 and 0.727.",
  r"intervals are $[0.230,0.292]$ for depth and $[0.403,0.654]$ for promotion; exceedance-share lower bounds are 0.560 and 0.726.")
R(r"There are 3 depth and 42 promotion cells with $|b_M|<0.05$, leaving 251 and 212 cells respectively after exclusion;",
  r"There are 2 depth and 42 promotion cells with $|b_M|<0.05$, leaving 247 and 207 of the 249 cells respectively after exclusion;")

# ---------------------------------------------------------------- 3.9
R(r"""On uncensored days, Pearson $\chi^2/\mathrm{df}$ has median 3.22. Censored NB2 converges in 16/254 cells (6\%); ZIP with intercept-only inflation in 9/254 (4\%), and with promotion-dependent inflation in 1/254 (0.4\%). These sparse fits limit alternative-family conclusions. Mixture-censoring sensitivity fits have convergence limitations, with convergence ranging from 87--168 of 254 cells at fixed mixture probabilities and 20 of 254 with an estimated probability. Sparse convergence also limits the overdispersed and zero-inflated comparisons. The headline gap remains conditional on the Poisson recovery model.""",
  r"""On uncensored days, Pearson $\chi^2/\mathrm{df}$ has median 3.22 (249 cells). Under the corrected solver contract the censored NB2 recovery is valid in 252/254 cells, ZIP in 238/254 (intercept-only inflation) and 237/254 (promotion-dependent inflation), and the mixture-censoring fits in 245/247/248 of 254 at fixed $\pi=0.95/0.85/0.75$ and 248/254 with $\pi$ estimated per cell. The previously reported counts (NB2 16/254, ZIP 9/254, mixtures 87--168/254 and 20/254) came from the earlier solver, whose tail floor and convergence flags were defective (Appendix~\ref{app:repro}). With each alternative as the recovery reference, the observed-sales-versus-recovery gaps satisfy the pre-fixed lower-bound rule for both coefficients (Appendix~\ref{app:recovery}). These are recovery-reference sensitivity checks, not matched alternative-family comparisons, and the headline gap remains conditional on the Poisson recovery model.""")

# ---------------------------------------------------------------- 3.10 E2
R(r"""Five estimators are compared on each draw: \textbf{naive} (raw observed sales), \textbf{G} (the manuscript's real, converged, full-row Poisson GLM, not a censoring-aware model), \textbf{M} (the paper's censoring-corrected recovery model), and two distributional alternatives, \textbf{NB2} and \textbf{MIX} (zero-inflated).""",
  r"""Five estimators are compared on each draw: \textbf{naive} (raw observed sales), \textbf{G} (the manuscript's real, converged, full-row Poisson GLM, not a censoring-aware model), \textbf{M} (the paper's censoring-corrected recovery model), and two distributional alternatives, \textbf{NB2} and \textbf{MIX} (zero-inflated). All fits are evaluated under the corrected validity rule; a fit without a valid estimate is counted as a failure, not used as an estimate. The E2 numbers use a reconstruction of the original inputs (Appendix~\ref{app:repro}): exact bit replay of the original shards was not possible; keys and true coefficients are identical for all 35,100 datasets and the generated data are identical for 28,985.""")
R(r"G (0.21--0.45) and naive (0.41--0.45) are both worse by construction", r"G (0.21--0.45) and naive (0.41--0.42) are both worse by construction")
R(r"($\alpha=0.50$, promo: $+0.022$, Holm $p=0.036$)", r"($\alpha=0.50$, promo: $+0.023$, Holm $p=0.036$)")
A4_OLD_START = r"\textbf{A4 (non-ignorable stockout): a demonstrated failure condition, not a general result.}"
A4_OLD_END = r"see \S\ref{sec:4-15} for the exact scope and the resulting recommendation."
A4_NEW = r"""\textbf{A4 (non-ignorable stockout): non-existence of the censored-Poisson estimate.} When the capacity ceiling shares the latent demand shock, this generator censors almost every day once $\rho>0$: the median censoring share is 76.7\% at $\rho=0$ but 99.9\% at $\rho=0.5$ and 99.95\% at $\rho=1.0$. With (almost) all rows censored the censored-Poisson likelihood has no finite maximum: the recession screen certifies structural non-existence in 2,080 of 2,700 datasets at $\rho=0.5$ and 2,442 of 2,700 at $\rho=1.0$ (3 at $\rho=0$). Counting every attempt, M has no valid estimate in 0.8\%, 77.6\% (2,094/2,700) and 90.5\% (2,443/2,700) of datasets at $\rho=0/0.5/1.0$; NB2 in 2.5\%, 86.3\% and 95.3\%; MIX in 1.5\%, 77.7\% and 90.5\% (Figure~\ref{fig:a4_sensitivity}). Non-existence, not finite-sample error of a computed estimate, is the failure mode demonstrated here. The originally reported A4 results treated most of these datasets as converged (the earlier solver flagged 4,352 of the 4,526 non-existence datasets as converged and returned arbitrary stopping points), which produced the large error medians and the 10 of 12 significant tests reported before this correction. On identical repeat IDs the earlier and corrected estimates agree, so the change comes from failure accounting, not from different estimates.

On the minority of datasets with a valid M fit, the comparison with naive and G is mixed (Table~\ref{tab:e2a4}; all 12 Holm-adjusted tests, 4,000-draw city bootstrap). Valid fits leave 6--13 simulation cells per comparison at $\rho>0$, against 23--27 at $\rho=0$. At $\rho=0.5$, M has larger depth error than naive ($+0.381$) and G ($+0.244$), both Holm $p<0.001$, while the promotion differences are not significant after adjustment. At $\rho=1.0$, M has \emph{smaller} promotion error than naive ($-0.142$, Holm $p<0.001$) and larger depth error than G ($+0.903$, $p<0.001$); the other two differences are not significant. At $\rho=0$, M has smaller promotion error than G ($-0.025$, $p<0.001$). In total 5 of the 12 tests are Holm-significant, with differing directions. Among valid M fits at $\rho=0.5/1.0$ whose two true coefficients both pass the $|\beta_{\text{true}}|\geq0.05$ filter ($n=444/200$), 26.4\%/31.0\% have at least one coefficient off by more than 100\% of its true value and 2.3\%/1.5\% have both. These valid-fit results condition on the existence of a finite MLE, which holds in a selected minority of datasets, so they are read together with the all-attempt failure rates; see \S\ref{sec:4-15} for the scope and the resulting recommendation."""
BLK.append((A4_OLD_START, A4_OLD_END, A4_NEW))

# ---------------------------------------------------------------- new subsection (controlled simulation), before Discussion
CS = r"""
\subsection{Controlled simulation: demand--capacity dependence at fixed censoring rates}
\label{sec:controlled}

\textit{[Insert Table 12 about here.]}

\noindent Figure~\ref{fig:controlled_sim} summarizes the results.

E2 varies the shared-shock strength and the censoring rate together, so under A4 the censoring rate rises to nearly 100\% and the estimator ceases to exist. A separate, fully synthetic experiment asks how the relative accuracy of censored Poisson changes when demand--capacity dependence changes while the expected censoring rate is held fixed. It follows a fixed protocol (version 1.0), is kept distinct from E2, and uses no FreshRetailNet-50K data. Two dependence levels $\eta\in\{0,0.6\}$ (a latent-normal correlation between demand and capacity) are crossed with three expected censoring rates $q\in\{0.20,0.60,0.95\}$, obtained by calibrating the capacity mean of each stratum so that the population probability $\Pr(D>C)$ equals $q$. Each of 1,000 replicate IDs is evaluated in all six conditions, with demand identical across conditions within an ID. A dataset has 2,400 observations in four strata (no promotion, and promotion at depth 0.05, 0.15 and 0.30) with marginally Poisson demand and $(\beta_0^\star,\beta_P^\star,\beta_d^\star)=(\log 20,0.20,1.0)$. G is the observed-sales Poisson fit, M the censored-Poisson fit (which ignores the dependence by design), and an oracle fitted to latent demand serves as a reference; all fits use the validity rule of Appendix~\ref{app:repro}. The loss $A$ is the absolute error of the predicted promotion lift $\exp(\hat\beta_P+d\hat\beta_d)-1$ in percentage points (pp), averaged over the three depths; coefficient errors $E_P$ and $E_d$ are relative. Within each ID and condition $B=A_{\text{M}}-A_{\text{G}}$; $\Delta_{\eta q}$ is the median of $B$ over the IDs $V_{\eta q}$ with both fits valid, and the primary statistic $\Theta_{0.60}$ is the median within-ID change in $B$ from $\eta=0$ to $\eta=0.6$ at $q=0.60$. Intervals are 95\% Monte Carlo intervals from 2,000 joint bootstrap resamples of replicate IDs. Design detail, failure and information diagnostics (Table A8), the oracle summary and descriptive outputs are in Online Resource 1.

All 6,000 datasets and 13,000 fits were attempted and none failed (no invalid G or M fit in any condition), so $|V_{\eta q}|=1{,}000$ throughout. The realized censoring share matches $q$ (means 0.2000, 0.5997--0.5999 and 0.9499--0.9500), but equal censoring does not mean equal information: the median smallest uncensored stratum count falls from 313 at $q=0.20$ to 16 at $q=0.95$. M has the smaller lift error in all six conditions (Table~\ref{tab:cs_main}): $\Delta_{\eta q}$ is $-0.50$, $-4.45$ and $-17.88$ pp at $q=0.20/0.60/0.95$ without dependence and $-0.16$, $-2.09$ and $-8.00$ pp with $\eta=0.6$, all with intervals below zero. Dependence makes the comparison less favorable to M: $\Theta_{0.60}=2.56$ pp (95\% interval $[2.45, 2.70]$, Monte Carlo standard error 0.068, $|W_{0.60}|=1{,}000$), with secondary values $\Theta_{0.20}=0.22$ and $\Theta_{0.95}=10.05$ pp. A positive $\Theta$ does not show that M is worse than G; here the median paired difference under $\eta=0.6$ is less than half of its $\eta=0$ value at each $q$. One coefficient-level result is adverse to M: at $q=0.20$ the paired depth relative error is larger for M ($\Delta E_d=+0.0030$ $[0.0011, 0.0053]$ at $\eta=0.6$; $+0.0018$ $[-0.0000, 0.0039]$ at $\eta=0$) and the promotion difference spans zero, although the lift error, which combines both coefficients, still favors M. At $q=0.60$ and $0.95$ both coefficient differences are negative. For reference, the oracle's median lift loss is 1.03 pp (10th--90th percentile 0.41--2.02).

These results describe a comparison under six specified synthetic conditions with one dependence structure. They do not establish the availability process in FreshRetailNet-50K, show that equal censoring implies equal identification, identify causal promotion effects, or establish optimal promotion policies. Together with E2 they separate two routes to failure: near-total censoring, under which the censored-Poisson estimate does not exist (E2, A4), and moderate dependence at a fixed censoring rate, under which M remains computable and, in this design, more accurate than G, with a smaller advantage.
"""
R("\n\\section{Discussion}\n\\label{sec:discussion}", CS + "\n\\section{Discussion}\n\\label{sec:discussion}")

# ---------------------------------------------------------------- Discussion
R(r"(the A4 result in \S\ref{sec:3-10} is a demonstrated failure under one specific shared-shock construction, not a law)",
  r"(the A4 result in \S\ref{sec:3-10} is a demonstrated non-existence failure under one specific shared-shock construction, not a law)")
R(r"Second, the E2 stress test shows the correction is not unconditionally safe: under non-ignorable stockout timing it can perform worse than reporting nothing corrected at all (\S\ref{sec:4-15}).",
  r"Second, the correction is not unconditionally available or safe: in the E2 stress test, strongly non-ignorable stockout timing censors nearly every day, so the censored-Poisson estimate usually does not exist, and where it exists its error relative to naive/G is mixed (\S\ref{sec:4-15}); in the controlled simulation, demand--capacity dependence at a fixed censoring rate reduces M's lift-error advantage without reversing it (\S\ref{sec:controlled}).")

# ---------------------------------------------------------------- Limitations
R(r"Mixture fits show similar gaps where they converge, with convergence limitations; they do not establish a bound",
  r"Mixture fits (valid in 245--248 of 254 cells) show similar gaps; they do not establish a bound")
R(r"and shows similar gaps for the fixed-$\pi$ variants where they converge (depth $0.2397$--$0.2681$; 87--168/254); the per-cell estimated-$\hat\pi$ variant converges in only 20/254 cells and gives a markedly different depth median there (0.7333), not a confirmation of the fixed-$\pi$ range.",
  r"and shows similar gaps (depth 0.253--0.265 at fixed $\pi$, valid in 245--248/254 cells; 0.260 with per-cell $\hat\pi$, valid in 248/254). The earlier estimated-$\hat\pi$ result (20 cells, depth 0.733) came from fits that are not valid under the corrected contract and is superseded.")
R(r"(width ratio 0.92--1.07 on the four matched Poisson statistics)", r"(width ratio 0.88--1.17 on the four matched Poisson statistics)")
R(r"(matched Poisson median gaps: wild lower bounds 0.226 (depth) / 0.405 (promo) $>0.2$; frac $\geq20\%$: 0.556 (depth) / 0.727 (promo) $>0.5$;",
  r"(matched Poisson median gaps: wild lower bounds 0.230 (depth) / 0.403 (promo) $>0.2$; frac $\geq20\%$: 0.560 (depth) / 0.726 (promo) $>0.5$;")
R(r"and it is a separate check from the eight pre-specified Holm-adjusted tests of \S\ref{sec:3-10}.",
  r"and it is a separate check from the twelve Holm-adjusted A4 tests of \S\ref{sec:3-10}.")
R(r"M's advantage over naive/G survives distributional misspecification and censoring-label noise (arms A0--A3), but fails under one specific non-ignorable-censoring construction (arm A4, \S\ref{sec:4-15}), where M, NB2, and MIX all become worse than not correcting at all.",
  r"M's advantage over naive/G survives distributional misspecification and censoring-label noise (arms A0--A3); under one specific non-ignorable-censoring construction (arm A4, \S\ref{sec:4-15}) the censored-Poisson estimate usually does not exist, and on the datasets where it exists its comparison with naive/G is mixed.")
R(r"The mixture fits leave gaps similar where they converge, while sparse NB2/ZIP convergence prevents a confirmatory robustness claim.",
  r"The NB2, ZIP and mixture refits are valid in 237--253 of 254 cells and keep the recovery-reference gaps above the 20\% threshold (Appendix~\ref{app:recovery}), but they are not matched comparisons.")

S415_START = r"\textbf{A demonstrated failure condition: M, NB2, and MIX all break under non-ignorable stockout timing (simulation E2, \S\ref{sec:3-10}).}"
S415_END = r"where ignorability is itself an unverified assumption that dual reporting is designed to surface, not assume away."
S415_NEW = r"""\textbf{A demonstrated failure condition: under strongly non-ignorable stockout timing the censored-Poisson estimate usually does not exist (simulation E2, \S\ref{sec:3-10}).}
\label{sec:4-15}

\S\ref{sec:3-10}'s simulation benchmark found M's distributional robustness to hold under misspecified demand distributions and censoring-label noise (A0--A3). A4 behaves differently. When the stockout capacity ceiling is driven by the same latent shock as demand, the tested generator censors nearly every day once $\rho>0$ (median censoring share 99.9\% and 99.95\% at $\rho=0.5/1.0$), and the censored-Poisson likelihood then typically has no finite maximum (structural non-existence in 2,080 and 2,442 of 2,700 datasets). This non-existence, rather than a finite-sample error of a computed estimate, is the failure mode the simulation demonstrates; NB2 and MIX fail at equal or higher rates. On the minority of datasets with a finite MLE, the M-versus-naive/G comparison is mixed in sign and significance (Table~\ref{tab:e2a4}).

\textbf{Scope, stated precisely.} This is a demonstrated failure under one specific generative construction (a shared latent shock linking demand and capacity), not a general law that ``all non-ignorable censoring breaks M.'' The earlier open question, whether the extreme A4 errors reflected bias or near-total censoring, is resolved by the per-dataset censoring shares: near-total censoring and the resulting non-existence explain them. Two things remain \emph{not} established: (1) whether real FreshRetailNet-50K stockouts have this shared-shock property; we have no instrument for latent demand, and the real panels (median stockout-day share 58.8\%, valid M in 249 of 254 cells) are far from the near-total censoring of A4 at $\rho>0$; (2) how non-ignorability affects M when censoring is not near-total. The controlled simulation (\S\ref{sec:controlled}) addresses (2) for one Gaussian-copula dependence at fixed censoring rates and finds a smaller advantage for M, not a reversal; other dependence structures remain untested.

\textbf{Resulting recommendation.} Because no validated pre-test for non-ignorable (as opposed to merely frequent) censoring is presented in this paper, we do not prescribe withholding M until one is run. Instead: report both the naive/G estimate and the M-corrected estimate side by side (dual reporting) as the default whenever the censoring mechanism's ignorability has not been independently verified; confirm that M has a finite, valid maximum (recession screen and stationarity on the original and standardized scales) before reading it, and report a non-existent or non-valid fit as such rather than substituting a number; and flag divergence between the two estimates as the diagnostic signal itself. The absence of a specific reason to suspect non-ignorable stockout timing (e.g., coordinated promotions timed to expected demand, supply shortages correlated with anticipated spikes) is not the same as having verified the assumption, so it does not on its own license substituting M unconditionally. This recommendation is narrower than \S\ref{sec:3-8}'s original unconditional-substitution claim: robustness to the distributional and label-noise misspecifications tested in A0--A3 stands on its own terms, but it does not extend to non-ignorable censoring (A4), where ignorability is itself an unverified assumption that dual reporting is designed to surface, not assume away."""
# the old block contains its own \label{sec:4-15}; replace block including label line
BLK.append((S415_START, S415_END, S415_NEW))

R(r"We have now refit the alternative ordering (censoring evaluated first, at the reduced covariate set; 254/254 converged) and report it in \S\ref{sec:3-4}: the total gap is order-invariant (exact telescoping identity) while the attribution is order-sensitive (censoring component $0.259\rightarrow0.378$ median for depth);",
  r"We have now refit the alternative ordering (censoring evaluated first, at the reduced covariate set; valid in 251/254 and 248/254 cells) and report it in \S\ref{sec:3-4}: the total gap is order-invariant (exact telescoping identity) while the attribution is order-sensitive (censoring component $0.259\rightarrow0.378$ median for depth);")

# ---------------------------------------------------------------- figures
R(r"""	\caption{Hour-of-day profile: mean per-hour promotion coefficient for the naive (observed) and recovered (M) estimators. Same data as Table~\ref{tab:hourly}.}""",
  r"""	\caption{Hour-of-day profile: mean per-hour promotion coefficient for the naive (observed) and recovered (M) estimators, 249 cells with a valid corrected M, corrected uncensored-day row mask. Same data as Table~\ref{tab:hourly}.}""")
R(r"""	\caption{E2 arm A4 (non-ignorable stockout): median relative error \emph{by estimator} (naive, G, M, NB2, MIX) as a function of shared-shock strength $\rho$, for the depth and promo coefficients. Sourced from the same E2 simulation as Table~\ref{tab:e2a4}, but a distinct statistic: this plots each estimator's own per-arm median relative error, not Table~\ref{tab:e2a4}'s paired M-vs-naive/G median \emph{difference} and its Holm-corrected significance test.}""",
  r"""	\caption{E2 arm A4 (non-ignorable stockout), corrected analysis. Left and middle: median relative error \emph{by estimator} (naive, G, M, NB2, MIX) among valid fits as a function of shared-shock strength $\rho$, for the depth and promo coefficients (log scale). Right: share of all attempted fits (2,700 per $\rho$) without a valid estimate; at $\rho=0.5/1.0$ most M, NB2 and MIX fits fail because the censored likelihood has no finite maximum, so the left and middle panels rest on a selected minority of datasets. Distinct from Table~\ref{tab:e2a4}'s paired M-vs-naive/G median \emph{difference} and its Holm-corrected test.}""")
R(r"""	\label{fig:a4_sensitivity}
\end{figure}
""", r"""	\label{fig:a4_sensitivity}
\end{figure}

\begin{figure}[p]
	\centering
	\includegraphics[width=\textwidth]{fig_controlled_sim.pdf}
	\caption{Controlled simulation (\S\ref{sec:controlled}), 1,000 replicate IDs $\times$ 6 conditions. (A) $\Delta_{\eta q}$, median paired M$-$G lift-error difference with 95\% Monte Carlo intervals; (B) distribution of within-ID differences $B$ at $q=0.60$ (median and middle 80\% of IDs: a distribution range, not an interval); (C) $\Theta_q$ with 95\% Monte Carlo intervals, $q=0.60$ primary; (D) share of IDs with jointly valid M and G fits, $100|V_{\eta q}|/1000$. Negative values favor M. Same data as Table~\ref{tab:cs_main}.}
	\label{fig:controlled_sim}
\end{figure}
""")

# ---------------------------------------------------------------- tables
R(r"""		Median                                                                  & \textbf{0.456}                                                      \\
		p10 / p25 / p75 / p90                                                   & 0.069 / 0.241 / 0.802 / 1.428                                       \\""",
  r"""		Median                                                                  & \textbf{0.455}                                                      \\
		p10 / p25 / p75 / p90                                                   & 0.068 / 0.236 / 0.802 / 1.437                                       \\""")
R(r"\caption{model-relative gap magnitude in mean promotion effects (P1a)}", r"\caption{model-relative gap magnitude in mean promotion effects (P1a), 249 cells with a valid corrected M}")
R(r"""		Unmatched benchmark                                                & depth    & 0.502          & ---           & 0.764           \\
		Unmatched benchmark                                                & promo    & 0.841          & ---           & 0.827           \\
		Covariate omission                                                 & depth    & 0.384          & $+$0.344      & 0.634           \\
		Covariate omission                                                 & promo    & 0.521          & $+$0.183      & 0.760           \\
		Distributional/link form                                           & depth    & 0.305          & $-$0.208      & 0.650           \\
		Distributional/link form                                           & promo    & 0.310          & $+$0.160      & 0.642           \\
		\shortstack[l]{Matched censored-versus-observed\\Poisson contrast} & depth    & \textbf{0.259} & $-$0.121      & 0.598           \\
		\shortstack[l]{Matched censored-versus-observed\\Poisson contrast} & promo    & \textbf{0.525} & $+$0.359      & 0.772           \\""",
  r"""		Unmatched benchmark                                                & depth    & 0.500          & ---           & 0.763           \\
		Unmatched benchmark                                                & promo    & 0.834          & ---           & 0.827           \\
		Covariate omission                                                 & depth    & 0.383          & $+$0.342      & 0.631           \\
		Covariate omission                                                 & promo    & 0.525          & $+$0.184      & 0.759           \\
		Distributional/link form                                           & depth    & 0.306          & $-$0.209      & 0.651           \\
		Distributional/link form                                           & promo    & 0.305          & $+$0.155      & 0.639           \\
		\shortstack[l]{Matched censored-versus-observed\\Poisson contrast} & depth    & \textbf{0.259} & $-$0.126      & 0.602           \\
		\shortstack[l]{Matched censored-versus-observed\\Poisson contrast} & promo    & \textbf{0.525} & $+$0.354      & 0.771           \\""")
R(r"\caption{Decomposition components on the common denominator $|b_{\text{M}}|$ (254 cells;", r"\caption{Decomposition components on the common denominator $|b_{\text{M}}|$ (249 cells with a valid corrected M;")
R(r"""			Coefficient (of 254 cells) & Flagged both & Unmatched only & Matched only & Flagged neither \\
			\midrule
			Depth                      & 128          & 66             & 24           & 36              \\
			Promotion                  & 171          & 39             & 25           & 19              \\""",
  r"""			Coefficient (of 249 cells) & Flagged both & Unmatched only & Matched only & Flagged neither \\
			\midrule
			Depth                      & 127          & 63             & 23           & 36              \\
			Promotion                  & 167          & 39             & 25           & 18              \\""")
R(r"between the unmatched (naive-reduced-vs-M) and full matched (glm-vs-M) comparisons, 254 cells.", r"between the unmatched (naive-reduced-vs-M) and full matched (glm-vs-M) comparisons, 249 cells with a valid corrected M.")
R(r"""		00   & -0.064                             \\
		06   & -0.012                             \\
		12   & -0.064                             \\
		18   & \textbf{-0.211}                    \\
		23   & -0.081                             \\""",
  r"""		00   & -0.089                             \\
		06   & -0.012                             \\
		12   & -0.046                             \\
		18   & -0.183                             \\
		20   & \textbf{-0.232}                    \\
		23   & -0.070                             \\""")
R(r"\caption{Hour-of-day profile}", r"\caption{Hour-of-day profile (mean naive$_h-$recovered$_h$, 249 cells, corrected row mask)}")
R(r"""		(0.0, 0.072]   & 458,533 & 2.238           & 2.588               & -0.349 & $[-0.637, -0.069]$     \\
		(0.072, 0.119] & 445,455 & 2.034           & 2.514               & -0.480 & $[-0.885, -0.080]$     \\
		(0.119, 0.171] & 472,383 & 2.057           & 2.499               & -0.442 & $[-0.810, -0.082]$     \\
		(0.171, 0.258] & 456,612 & 2.134           & 2.588               & -0.454 & $[-0.810, -0.101]$     \\
		(0.258, 1.0]   & 462,018 & 2.425           & 2.893               & -0.467 & $[-0.835, -0.106]$     \\""",
  r"""		(0.0, 0.071]   & 435,819 & 2.243           & 2.596               & -0.353 & $[-0.643, -0.071]$     \\
		(0.071, 0.122] & 436,437 & 2.042           & 2.523               & -0.481 & $[-0.881, -0.089]$     \\
		(0.122, 0.173] & 435,898 & 2.060           & 2.507               & -0.447 & $[-0.816, -0.086]$     \\
		(0.173, 0.263] & 438,740 & 2.153           & 2.603               & -0.450 & $[-0.800, -0.103]$     \\
		(0.263, 1.0]   & 437,762 & 2.434           & 2.903               & -0.469 & $[-0.840, -0.106]$     \\""")
R(r"\caption{Dose-response across discount-depth quintiles. CI:", r"\caption{Dose-response across discount-depth quintiles (249 cells with a valid corrected M). CI:")
R(r"""		Statistic (converged cells)                               & Value                                                                    \\
		\midrule
		Converged                                                 & \textbf{15 / 20} (Gauss--Newton FE step completed)                       \\
		$|bias_{\text{B}}|/|naive_{\text{B}}|$: median (min--max) & \textbf{1.274} (0.242--7.344)                                            \\
		Non-converged (flagged \& excluded from F-based claims)   & 5 cells: 0$\times$5, 0$\times$16, 0$\times$18, 0$\times$28, 12$\times$20 \\
		\bottomrule
	\end{tabular}
	\caption{Fixed-effect sensitivity (F-model), 20 cells with largest promo $\times$ stockout overlap}""",
  r"""		Statistic                                                 & Value                                                                    \\
		\midrule
		Jointly valid                                             & \textbf{20 / 20} (joint Newton, 6--10 iterations)                         \\
		$|bias_{\text{B}}|/|naive_{\text{B}}|$: median (min--max) & \textbf{0.600} (0.071--4.048)                                            \\
		All-censored store--days (no recoverable demand)          & 111--9,653 per cell; their rows (1.3--42.8\%) excluded                   \\
		\bottomrule
	\end{tabular}
	\caption{Fixed-effect sensitivity (F-model, corrected joint fit), 20 cells with largest promo $\times$ stockout overlap; naive B on the same rows}
	\label{tab:fmodel}""")
R(r"""			B: store--day FE (sensitivity)                & -2.7933/hr & -4.4982/hr                                                   & +1.7049/hr          & 61.0\%           \\
			M / F model $\beta_{\text{promo}}$ (log link) & ---        & -0.4211 / -0.4754 ($\approx$ -34\% / -38\% of latent demand) & ---                 & ---              \\""",
  r"""			B: store--day FE (sensitivity)                & -2.8160/hr & -4.3786/hr                                                   & +1.5626/hr          & 55.5\%           \\
			M / F model $\beta_{\text{promo}}$ (log link) & ---        & -0.4211 / -0.4765 ($\approx$ -34\% / -38\% of latent demand) & ---                 & ---              \\""")

E2D_OLD_START = r"""		A0 (self-check)           & depth & 0.210"""
E2D_OLD_END = r"""is robust across both realizations and is the formally tested claim (Table~\ref{tab:e2a4}).}"""
E2D_NEW = r"""		A0 (self-check)           & depth & 0.209 & 0.064          & 0.062          & 0.416 & 0.064          \\
		A0 (self-check)           & promo & 0.445 & 0.092          & 0.091          & 0.408 & 0.092          \\
		A1 NB2-demand             & depth & 0.285 & 0.157          & 0.168          & 0.340 & 0.130          \\
		A1 NB2-demand             & promo & 0.514 & 0.195          & 0.194          & 0.461 & 0.195          \\
		A2 ZIP-demand             & depth & 0.214 & 0.130          & \textbf{0.059} & 0.314 & 0.125          \\
		A2 ZIP-demand             & promo & 0.458 & 0.155          & \textbf{0.086} & 0.496 & 0.155          \\
		A3 label noise            & depth & 0.219 & 0.110          & 0.108          & 0.400 & 0.095          \\
		A3 label noise            & promo & 0.491 & 0.138          & 0.112          & 0.418 & 0.129          \\
		A4 non-ignorable$^{a}$    & depth & 0.170 & 0.132          & 0.174          & 0.178 & 0.097          \\
		A4 non-ignorable$^{a}$    & promo & 0.210 & 0.127          & 0.129          & 0.170 & 0.107          \\
		A5 combined               & depth & 0.166 & 0.166          & 0.190          & 0.176 & 0.197          \\
		A5 combined               & promo & 0.163 & 0.195          & 0.199          & 0.169 & 0.277          \\
		\bottomrule
	\end{tabular}

	\footnotesize\textit{Corrected analysis on the reconstructed E2 inputs (Appendix~\ref{app:repro}): median over simulation cells of the per-cell median relative error among valid fits, with the $|\beta_{\text{true}}|\geq0.05$ filter on the relevant coefficient. Fits without a valid estimate are excluded from the medians and counted as failures. $^{a}$The pooled A4 row is dominated by $\rho=0$: at $\rho=0.5/1.0$ M, NB2 and MIX have no valid fit in 77.6--95.3\% of datasets because the censored likelihood has no finite maximum (Figure~\ref{fig:a4_sensitivity}, Table~\ref{tab:e2a4}). The values published before the correction (e.g.\ A4 M 0.825/1.271) counted such non-existence fits as converged estimates and are superseded.}"""
BLK.append((E2D_OLD_START, E2D_OLD_END, E2D_NEW))

E2A_OLD_START = r"""	\caption{E2 arm A4: M vs.\ naive/G, city-block bootstrap median difference (95\% CI), by $\rho$ and coefficient. Positive = M has larger error than the comparator.}"""
E2A_OLD_END = r"""		1.0    & depth       & \textbf{+0.70 [0.65, 0.99]}                  & \textbf{+0.76 [0.71, 0.97]}                             & $<0.001$ \\
		\bottomrule
	\end{tabular}"""
E2A_NEW = r"""	\caption{E2 arm A4: M vs.\ naive/G on valid paired fits, city-block bootstrap (4,000 draws) median difference in relative error (95\% CI) and Holm-adjusted $p$ over the 12-test family, by $\rho$ and coefficient. Positive = M has larger error than the comparator; bold = Holm $p<0.05$. Cells: simulation cells with valid paired fits. Share of all attempted M fits without a valid estimate: 0.8\%/77.6\%/90.5\% at $\rho=0/0.5/1.0$.}
	\label{tab:e2a4}
	\resizebox{\textwidth}{!}{%
	\begin{tabular}{@{}ccccccc@{}}
		\toprule
		$\rho$ & Coef  & Cells & M $-$ naive (95\% CI)                     & Holm $p$ & M $-$ G (95\% CI)                         & Holm $p$ \\
		\midrule
		0.0    & promo & 23    & $-$0.003 [$-$0.010, 0.009]               & 1.000    & \textbf{$-$0.025 [$-$0.040, $-$0.013]}   & $<0.001$ \\
		0.0    & depth & 27    & $+$0.001 [$-$0.011, 0.029]               & 1.000    & $-$0.017 [$-$0.036, $-$0.003]            & 0.123    \\
		0.5    & promo & 10    & $+$0.185 [0.020, 0.244]                  & 0.219    & $+$0.166 [$-$0.340, 0.396]               & 1.000    \\
		0.5    & depth & 13    & \textbf{$+$0.381 [0.205, 1.780]}         & $<0.001$ & \textbf{$+$0.244 [0.190, 1.178]}         & $<0.001$ \\
		1.0    & promo & 6     & \textbf{$-$0.142 [$-$0.225, $-$0.024]}   & $<0.001$ & $+$0.044 [$-$0.865, 0.241]               & 1.000    \\
		1.0    & depth & 8     & $+$0.891 [$-$0.110, 17.896]              & 0.365    & \textbf{$+$0.903 [0.050, 17.796]}        & $<0.001$ \\
		\bottomrule
	\end{tabular}%
	}"""
BLK.append((E2A_OLD_START, E2A_OLD_END, E2A_NEW))

R(r"""		P1a median model-relative gap ratio & 0.456          & 0.403                                  \\
		P1a fraction $\geq$20\%             & 0.803          & 0.749                                  \\
		Matched depth median gap            & 0.259          & 0.228                                  \\
		Matched depth fraction $\geq$20\%   & 0.598          & 0.548                                  \\
		Matched promo median gap            & 0.525          & 0.396                                  \\
		Matched promo fraction $\geq$20\%   & 0.772          & 0.710                                  \\
		Matched depth sign-flip rate        & 0.110          & 0.065                                  \\
		Matched promo sign-flip rate        & 0.134          & 0.094                                  \\""",
  r"""		P1a median model-relative gap ratio & 0.455          & 0.388                                  \\
		P1a fraction $\geq$20\%             & 0.799          & 0.745                                  \\
		Matched depth median gap            & 0.259          & 0.230                                  \\
		Matched depth fraction $\geq$20\%   & 0.602          & 0.552                                  \\
		Matched promo median gap            & 0.525          & 0.396                                  \\
		Matched promo fraction $\geq$20\%   & 0.771          & 0.708                                  \\
		Matched depth sign-flip rate        & 0.108          & 0.059                                  \\
		Matched promo sign-flip rate        & 0.137          & 0.096                                  \\""")
R(r"Approximate percentile-bootstrap coverage, not a finite-sample guarantee; a separate check from the eight pre-specified Holm-adjusted tests of \S\ref{sec:3-10}.}",
  r"Approximate percentile-bootstrap coverage, not a finite-sample guarantee; a separate check from the twelve Holm-adjusted A4 tests of \S\ref{sec:3-10}. 249 cells with a valid corrected M.}")

# controlled-simulation core table appended after bonf8 (Table 12)
CS_TAB = open(os.path.expanduser("~/papers/jams-frn50k/latex/cs_main_table.tex")).read()
R("""	\\label{tab:bonf8}
\\end{table}
""", """	\\label{tab:bonf8}
\\end{table}

""" + CS_TAB)

# ---------------------------------------------------------------- appendix
REPRO_OLD_START = r"""\subsection*{Reproducibility}
Pilot criteria were fixed before the main analysis (unregistered)."""
REPRO_OLD_END = r"""is retained only as a historical record)."""
REPRO_NEW = r"""\subsection*{Reproducibility and correction note}
\label{app:repro}
Pilot criteria were fixed before the main analysis (unregistered). The E2 misspecified-DGP simulation (\S\ref{sec:3-10}) has a deterministic reproduction entry point, \texttt{main/11\_e2\_run\_all.py} (seed 42 throughout; city-cluster bootstrap $n_{\text{boot}}=4000$), which reruns the analysis/bootstrap/descriptive scripts on the simulation shards; the A4 fit-level error-rate statistics (26.4\%/31.0\%/2.3\%/1.5\%, $n=444/200$) are computed from the same intermediate by a separate script in the correction package. The real-data recovery-model sensitivity check (\S\ref{sec:3-9}) is reproduced by \texttt{main/07\_pending\_3.py} $\to$ \texttt{08\_pending\_final.py} $\to$ \texttt{10\_fix\_posthoc.py}; model-preference counts are taken over jointly valid cells.

\textbf{Correction of the existing analysis.} An audit of the estimation code found numerical defects, corrected before the results in this version were computed: (i) the censored-Poisson upper tail was clipped at $10^{-300}$ (and $\lambda$ at $[10^{-8},10^{6}]$); it is now computed strictly in the log domain without a floor, as are the NB2 and mixture tails; (ii) the censored-row curvature had the wrong sign in the score/curvature routine used by the fixed-effect fit; (iii) convergence had been declared from solver success alone; a fit is now valid only if it passes a recession (linear-programming) screen for structural non-existence, a rank check, and a projected-gradient check ($\leq10^{-6}$ on the original and the column-standardized scale), and non-valid fits are reported, never used; (iv) the uncensored-day row mask of the hour-of-day apportionment selected the wrong rows; (v) the store--day fixed-effect fit had no joint convergence check and is now a joint Newton fit. The solver contract (average negative log likelihood; L-BFGS-B with at most 5,000 iterations, 50,000 evaluations and 50 line-search steps; three fixed starts; lowest objective among valid starts) was frozen before any production fit, and 26/26 numerical checks pass (tails against direct summation, finite-difference gradients and Hessians, recovery on known truth, multistart agreement). On the 249 valid empirical cells the corrected and earlier M coefficients differ by a median of $2.1\times10^{-6}$ (promotion) and $1.5\times10^{-5}$ (depth), so the empirical changes come mainly from excluding the five cells without a valid fit. The E2 benchmark is a documented reconstruction: replaying the unmodified generator reproduces keys, true coefficients and anchors for all 35,100 datasets and the generated data for 28,985; exact bit replay of the other 6,115 was not possible because the generator calls the earlier, floating-point-sensitive M fit. All E2 numbers are computed on the frozen reconstructed inputs. A correction package (\texttt{repair\_log.md}; \texttt{fit\_diagnostics.csv} with every attempted fit and its status; \texttt{before\_after.csv} with each published and corrected value) is included in the replication archive."""
BLK.append((REPRO_OLD_START, REPRO_OLD_END, REPRO_NEW))

HOUR_OLD_START = r"""Cell-mean model-relative gap by hour of day (naive$_h$ - recovered$_h$, averaged across the 254 cells):"""
HOUR_OLD_END = r"""the naive estimate understates the latent effect by $\approx$30\% in the cell-average sense.

\end{itemize}"""
HOUR_NEW = r"""Cell-mean model-relative gap by hour of day (naive$_h$ - recovered$_h$, averaged across the 249 cells with a valid corrected M, corrected uncensored-day row mask):

\begin{itemize}

	\item The model-relative gap is negative at 22 of 24 hours (naive below recovered in all but 07:00 and 08:00), i.e. the naive estimate understates the latent promotion effect at most hours, with two marginal exceptions.

	\item It peaks in the evening (top-5 $|bias|$ hours: 20, 17, 16, 18, 19), consistent with late-day inventory depletion on promotion days.

	\item Day totals (cell averages): naive 4.68 vs. recovered 6.69 units/day: the naive estimate understates the latent effect by $\approx$30\% in the cell-average sense.

	\item The originally reported profile (254 cells; top-5 hours 17, 18, 16, 19, 15; one exception at 07:00) used a row mask that selected the wrong days for the uncensored-day profile. With the mask fixed, the recovered-demand peak hour changes in 160 of 254 cells and the gap peak in 178 of 254; daily totals are unchanged.

\end{itemize}"""
BLK.append((HOUR_OLD_START, HOUR_OLD_END, HOUR_NEW))

R(r"Both estimator pairs (A and B) agree on the direction and the order of magnitude (57--61\%)", r"Both estimator pairs (A and B) agree on the direction and the order of magnitude (57.2\% and 55.5\%)")

FE_OLD_START = r"""Store--day fixed-effect variant (alternating MLE) on the 20 cells"""
FE_OLD_END = r"""reported as a sensitivity check where it converges."""
FE_NEW = r"""Store--day fixed-effect variant on the 20 cells with the largest promo $\times$ stockout overlap; naive B (store--day within) vs. recovered F on the same rows.

\textbf{Correction note.} The originally reported F results came from an alternating algorithm whose censored-row curvature had the wrong sign and which had no joint convergence check; in 5 of the 20 cells the offset step stopped after one iteration and was reported as non-converged. Those results (15/20 converged; median ratio 1.274; e.g.\ cell 0$\times$5 recovered F 217.8 and cell 0$\times$28 $-528.6$) are superseded. The corrected joint Newton fit is valid in all 20 cells (6--10 iterations, joint stationarity $\leq10^{-6}$). Only promo, depth and activity vary within a store--day; the other covariates are absorbed by the offsets. Store--days on which every row is censored (111--9,653 per cell) have an infinite offset and carry no recoverable demand; their rows are excluded from both naive B and recovered F. On this basis cell 0$\times$5 gives recovered F 10.48 vs.\ naive 9.13, and cell 0$\times$28 $-14.06$ vs.\ $-5.40$; the median $|bias_{\text{B}}|/|naive_{\text{B}}|$ is 0.600 (0.071--4.048). The headline still rests on M, with F as a sensitivity check."""
BLK.append((FE_OLD_START, FE_OLD_END, FE_NEW))

R(r"Recovery alternatives were attempted on 254 cells (seed 42); failed optimizer iterates are not evidence of robustness.",
  r"Recovery alternatives were refitted on 254 cells under the corrected solver contract (seed 42); fits that are not valid are reported and never used.")
R(r"Completed: M converged in 254/254 cells; matched observed-sales comparison reported.",
  r"Completed: corrected M valid in 249/254 cells (3 nonconvergence, 1 unidentified, 1 structural non-existence); matched comparison reported on 249.")
R(r"Completed: reduced-spec recovery converged in 254/254 cells.", r"Completed: reduced-spec recovery valid in 251/254 (depth) and 248/254 (promotion) cells.")
R(r"Attempted: 16/254 converged (6\%); sparse convergence prevents a robustness conclusion.", r"Completed: valid in 252/254 cells; preferred over M by AIC/BIC/LRT in all 249 jointly valid cells.")
R(r"Attempted: 9/254 converged with intercept-only inflation (4\%); 1/254 with promotion-dependent inflation (0.4\%).", r"Completed: valid in 238/254 cells with intercept-only inflation and 237/254 with promotion-dependent inflation.")
R(r"Attempted: fixed $\pi=0.95/0.85/0.75$ converged in 168/117/87 cells; estimated $\hat\pi$ in 20/254. Similar gaps where converged, with convergence limitations.",
  r"Completed: fixed $\pi=0.95/0.85/0.75$ valid in 245/247/248 cells; estimated $\hat\pi$ in 248/254. Similar gaps; M preferred by AIC/BIC/LRT in every jointly valid cell.")
R(r"Attempted: recorded convergence counts 153/112/81/70; recovery-reference sensitivity only.", r"Completed: valid in 250/251/251/253 cells; recovery-reference sensitivity only.")
R(r"Attempted on 20 selected cells: 15 converged, 5 failed and were excluded from F-based claims.", r"Completed on 20 selected cells: jointly valid in 20/20 after the correction; all-censored store--days counted, not fitted.")
R(r"""Completed (simulation benchmark E2, \S\ref{sec:3-10}--\S\ref{sec:4-15}): M, NB2, and MIX are all robust to distributional misspecification and label noise (A0--A3); under non-ignorable, shared-shock stockout timing (A4), M, NB2, and MIX all have higher median error than naive/G. Formally hypothesis-tested only for M (vs.\ naive/G): 8/8 Holm-significant at $\rho=0.5/1.0$ (full $\rho=0/0.5/1.0$ grid: 10/12). Scoped to the simulated construction, not a general claim about real FreshRetailNet-50K stockouts.""",
  r"""Completed (E2, \S\ref{sec:3-10}--\S\ref{sec:4-15}; reconstructed inputs): M, NB2, and MIX robust to distributional misspecification and label noise (A0--A3). Under A4 with $\rho>0$ the censored-Poisson MLE does not exist in most datasets (M invalid in 77.6\%/90.5\% of attempts); on valid fits 5 of 12 Holm-adjusted M-vs-naive/G tests are significant, in both directions. Scoped to the simulated construction. \\
	Controlled simulation                              & Completed (\S\ref{sec:controlled}; Online Resource 1): 6,000 datasets, 13,000 fits, 0 failures; $\Theta_{0.60}=2.56$ pp $[2.45, 2.70]$.""")
R(r"filtered gap statistics; 3 depth / 42 promo cells have $|b_M|<0.05$.", r"filtered gap statistics; 2 depth / 42 promo cells have $|b_M|<0.05$.")
R(r"which motivated checking NB2 in the first place; the low convergence rate is itself the finding for that check, not a nuisance to average over.",
  r"which motivated checking NB2 in the first place; NB2 is valid in 252/254 cells and is the data-preferred family (below).")

CONV_OLD_START = r"""\textbf{Converged-subset check (the cells that actually fit).}"""
CONV_OLD_END = r"""Every gap remains a model-relative quantity (\S\ref{sec:2-4}). The specifications were fixed before the fits:"""
CONV_NEW = r"""\textbf{Gaps under the alternatives (valid fits).} Gaps are median $|b_{\text{G}}-b_{\text{alt}}|/|b_{\text{alt}}|$ with Rademacher wild-cluster 95\% intervals over the 18 cities, on the cells where each alternative is valid. NB2 (252 cells): depth 0.445 [0.361, 0.530], share $\geq$20\% 0.754; promotion 0.615 [0.520, 0.704], 0.829. ZIP with intercept-only inflation (238 cells): depth 0.288 [0.245, 0.330], 0.656; promotion 0.598 [0.520, 0.677], 0.794. Mixture censoring at fixed $\pi=0.95/0.85/0.75$: depth 0.253/0.265/0.261 (lower bounds 0.224--0.229, shares 0.589--0.599), promotion 0.510/0.509/0.501 (lower bounds 0.394--0.400, shares 0.761--0.770); with $\hat\pi$: depth 0.260 [0.230, 0.290], promotion 0.524 [0.408, 0.639]. The NB2-reference depth gap exceeds the Poisson matched gap partly because the observed-sales glm is Poisson while the reference is NB2, so this comparison also changes the distribution. The earlier converged-subset results (16 NB2 cells, depth interval [0.175, 0.758]; 9 ZIP cells, depth median 0.189 and promotion median 1.72) came from fits that are not valid under the corrected contract and are superseded. Every gap remains a model-relative quantity (\S\ref{sec:2-4}). The specifications were fixed before the fits:"""
BLK.append((CONV_OLD_START, CONV_OLD_END, CONV_NEW))

DEC_OLD_START = r"""\textbf{Decision rule and what it actually shows.}"""
DEC_OLD_END = r"""a specification diagnostic, not a claim about the Poisson-M gap's size, for the convergence reasons above."""
DEC_NEW = r"""\textbf{Decision rule and what it shows.} The 25.9\%/52.5\% medians would be said to \emph{survive} an alternative if, under it, the median gap is at least 20\%, the share of cells above 20\% is at least 50\%, and the lower bound of the cluster interval is above 20\%, computed on the cells where that alternative is valid. Under the corrected fits every alternative satisfies the rule for both coefficients (lowest lower bound 0.224 for depth and 0.394 for promotion; lowest share 0.589). This revises the earlier statement that NB2 and ZIP convergence was too sparse to apply the rule; that sparsity came from the earlier solver. The rule is a recovery-reference sensitivity check: it shows that the gap between the observed-sales glm and a censoring-aware recovery is not specific to the Poisson recovery family, not that any recovery is closer to true demand, and it is not a matched alternative-family comparison (observed-sales NB2 fits are not recorded). The recommendation is to report both estimates conditional on the recovery model (\S\ref{sec:4-13}).

\textbf{Model preference (run).} Among jointly valid cells, the mixture-censoring specifications are not data-preferred: M wins AIC/BIC/LRT in all 242/245/244 comparable cells at $\pi=0.95/0.85/0.75$ and in all 244 with $\hat\pi$ (median $\Delta$AIC $+226$/$+718$/$+1{,}277$ and $+3.4$). Zero inflation is small ($\hat{\psi}$ median 0.004), although ZIP is preferred over M by AIC in 231 of 237 comparable cells (median $\Delta$AIC $-128$). Overdispersion (NB2-type) is the data-preferred family: AIC/BIC/LRT prefer NB2 in 249/249 cells (median $\Delta$AIC $-4{,}497$) and NB2+mixture in 245/247, 221/247, 192/248 and 249/249 cells at $\pi=0.95/0.85/0.75$ and $\hat\pi$. This is a specification diagnostic, not a claim about the size of the Poisson-M gap."""
BLK.append((DEC_OLD_START, DEC_OLD_END, DEC_NEW))

# ---------------------------------------------------------------- apply token-level replacements
bad = []
for old, new, n in E:
    c = src.count(old)
    if c != n:
        bad.append((c, n, old[:90]))
        continue
    src = src.replace(old, new)
if bad:
    for b in bad:
        print("COUNT MISMATCH", b)
    sys.exit(1)
for a, b, new in BLK:
    assert src.count(a) == 1 and src.count(b) >= 1, a[:60]
    i0 = src.index(a); i1 = src.index(b, i0) + len(b)
    src = src[:i0] + new + src[i1:]

def census(t):
    return Counter(len(m.group(1)) for m in re.finditer(r"(\\+)\s*$", t, re.M))
print("trailing-backslash census before:", dict(census(orig)), "after:", dict(census(src)))
assert census(src).get(1, 0) == census(orig).get(1, 0), "single-backslash line count changed"
for env in ("table", "tabular", "figure", "itemize", "enumerate", "longtable"):
    assert src.count(f"\\begin{{{env}}}") == src.count(f"\\end{{{env}}}"), env
open(P, "w").write(src)
print("OK: applied", len(E), "replacements + block rewrites")
