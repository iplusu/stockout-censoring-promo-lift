r"""P7 page-budget pass 6: appendix wording."""
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


rep(r"""	\item The originally reported profile (top-5 hours 17, 18, 16, 19, 15; one exception at 07:00) used a wrong row mask; fixing it moves the recovered peak hour in 160 and the gap peak in 178 of 254 cells; daily totals are unchanged.

""", "")
rep(r"""\textbf{Correction note.} The originally reported F results came from an alternating algorithm with a wrong-signed censored-row curvature and no joint convergence check (5 of 20 cells stopped after one iteration). Those results (15/20 converged; median ratio 1.274; cell 0$\times$5 recovered F 217.8, cell 0$\times$28 $-528.6$) are superseded. The corrected joint Newton fit is valid in all 20 cells (6--10 iterations). Store--days with every row censored (111--9,653 per cell) have an infinite offset and no recoverable demand; their rows are excluded from naive B and recovered F. Cell 0$\times$5 now gives recovered F 10.48 vs.\ naive 9.13, cell 0$\times$28 $-14.06$ vs.\ $-5.40$; median $|bias_{\text{B}}|/|naive_{\text{B}}|$ 0.600 (0.071--4.048). The headline rests on M, with F as a sensitivity check.""",
    r"""\textbf{Correction note.} The originally reported F results (15/20 converged; median ratio 1.274; cell 0$\times$5 recovered F 217.8, cell 0$\times$28 $-528.6$) came from an alternating algorithm with a wrong-signed censored-row curvature and no joint convergence check, and are superseded. The corrected joint Newton fit is valid in all 20 cells; rows of all-censored store--days (infinite offset, no recoverable demand) are excluded from naive B and recovered F. Cell 0$\times$5 now gives 10.48 vs.\ naive 9.13, cell 0$\times$28 $-14.06$ vs.\ $-5.40$ (Table~\ref{tab:fmodel}). The headline rests on M, with F as a sensitivity check.""")
rep(r"""(i) the censored upper tail was floored at $10^{-300}$; it is now computed strictly in the log domain (also for NB2 and mixtures); (ii) the censored-row curvature had the wrong sign; (iii) convergence had been declared from solver success alone; a fit is now valid only if a recession screen finds no structural non-existence, the design is identified, and the projected gradient is $\leq10^{-6}$ on the original and standardized scales, and non-valid fits are reported, never used; (iv) the uncensored-day row mask of the hourly apportionment selected the wrong rows; (v) the store--day fixed-effect fit had no joint convergence check.""",
    r"""(i) the censored upper tail was floored at $10^{-300}$ and is now computed strictly in the log domain (also for NB2 and mixtures); (ii) the censored-row curvature had the wrong sign; (iii) convergence had been declared from solver success alone, whereas a fit is now valid only if a recession screen finds no structural non-existence, the design is identified, and the projected gradient is $\leq10^{-6}$ on the original and standardized scales (non-valid fits are reported, never used); (iv) the hourly apportionment used a wrong uncensored-day row mask; (v) the store--day fixed-effect fit lacked a joint convergence check.""")
rep(r"""Zero inflation is small ($\hat{\psi}$ median 0.004), though ZIP is AIC-preferred in 231 of 237 cells (median $\Delta$AIC $-128$). NB2 is the data-preferred family (249/249 cells, median $\Delta$AIC $-4{,}497$; NB2+mixture 245/247, 221/247, 192/248 and 249/249 at $\pi=0.95/0.85/0.75$ and $\hat\pi$), a specification diagnostic, not a claim about the size of the Poisson-M gap.""",
    r"""Zero inflation is small ($\hat{\psi}$ median 0.004), though ZIP is AIC-preferred in 231 of 237 cells (median $\Delta$AIC $-128$). NB2 is data-preferred (249/249 cells, median $\Delta$AIC $-4{,}497$; NB2+mixture 245/247, 221/247, 192/248, 249/249), a specification diagnostic only.""")
open(P, "w").write(s)
print("OK")
