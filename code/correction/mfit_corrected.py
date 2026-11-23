"""Corrected censored-count fitters for the existing-analysis repair (instructions §2-3).

One numerical kernel for M (censored Poisson), NB2 (censored negative binomial, log-theta) and MIX
(censored zero-inflated Poisson, logit-pi), general design X (n x p) and optional offset.
  * strict right tail Pr(Y > K) in log space, no probability floor, no lambda/ratio clipping
      Poisson: gammainc(K+1, mu) if > 1e-250 else logpmf(K+1) + log(1 + sum_m prod mu/(K+j))   [ported
      from controlled_sim/sim.py, validated there 30/30]
      NB2:     betainc(K+1, theta, 1-p) if > 1e-250 else logpmf(K+1) + log(1 + sum_m prod (K+j-1+theta)/(K+j) (1-p))
  * average negative log likelihood (objective / n), analytic beta scores; NB2 log-theta score by
    centered difference of the same strict objective (no closed form); M analytic curvature with the
    cancellation-free far-tail form mu - (K+1)(T-1)/T.
  * finite-maximum screen generalized to p columns (recession LP), identification checked separately.
Validity (fixed before production): solver success, finite objective/coefficients, sup-norm of the
average-scale gradient <= GTOL on the original scale AND on the column-standardized scale.
"""
import math, warnings
import numpy as np
from scipy import optimize, special

TAIL_SWITCH = 1e-250
GTOL = 1e-6                 # validity threshold (both scales)
SOLVER_GTOL = 1e-9          # tighter internal stop so the validity check is meaningful
THETA_BOUNDS = (None, math.log(1e4))   # same model definition as legacy fit_nb2
PI_LOGIT_BOUNDS = (-8.0, 8.0)          # same model definition as legacy fit_zip


# ------------------------------------------------------------------ strict tails
def _pois_series(k, mu, tol=1e-17):
    lp = -mu + (k + 1) * math.log(mu) - math.lgamma(k + 2)
    t = s = 1.0; j = 2
    while True:
        r = mu / (k + j); t *= r; s += t; j += 1
        if r < 0.5 and t < tol * s:
            return lp + math.log(s)


def pois_logsf(k, mu):
    k = np.asarray(k, float); mu = np.asarray(mu, float)
    shape = np.broadcast(k, mu).shape
    k, mu = np.broadcast_to(k, shape).ravel(), np.broadcast_to(mu, shape).ravel()
    with np.errstate(divide='ignore', invalid='ignore'):
        S = special.gammainc(k + 1.0, mu); out = np.log(S)
    for i in np.flatnonzero(~(S > TAIL_SWITCH) & np.isfinite(mu) & (mu > 0)):
        out[i] = _pois_series(k[i], mu[i])
    return out.reshape(shape)


def pois_logpmf(k, mu):
    with np.errstate(divide='ignore', invalid='ignore'):
        return k * np.log(mu) - mu - special.gammaln(k + 1.0)


def _nb_logpmf(k, mu, th):
    """NB2 log pmf without cancellation at large theta (algebraically identical to
    lgam(k+th)-lgam(th)-lgam(k+1) + th log p + k log(1-p), p = th/(th+mu)):
        sum_{j<k} log1p(j/th) - lgam(k+1) - (th+k) log1p(mu/th) + k log mu
    The naive form loses ~1e-11 per row at th=1e4 (lgam(1e4) ~ 8.2e4), enough to stall the line search."""
    k = np.asarray(k, float); mu = np.asarray(mu, float)
    with np.errstate(divide='ignore', invalid='ignore'):
        kmax = int(np.max(k)) if k.size else 0
        L = np.concatenate([[0.0], np.cumsum(np.log1p(np.arange(kmax) / th))])
        return L[k.astype(np.int64)] - special.gammaln(k + 1.0) - (th + k) * np.log1p(mu / th) + k * np.log(mu)


def _nb_series(k, mu, th, tol=1e-17):
    q = mu / (th + mu)
    lp = float(_nb_logpmf(np.array([k + 1.0]), np.array([mu]), th)[0])
    t = s = 1.0; m = k + 1.0
    while True:
        r = (m + th) / (m + 1.0) * q          # f(m+1)/f(m)
        t *= r; s += t; m += 1.0
        if r < 1.0 - 1e-12 and t < tol * s and t * r / (1 - r) < tol * s:
            return lp + math.log(s)
        if m - k > 5e6:                        # ponytail: hard cap, flagged by caller as nonfinite if hit
            return float('nan')


def nb_logsf(k, mu, th):
    k = np.asarray(k, float); mu = np.asarray(mu, float)
    shape = np.broadcast(k, mu).shape
    k, mu = np.broadcast_to(k, shape).ravel(), np.broadcast_to(mu, shape).ravel()
    with np.errstate(divide='ignore', invalid='ignore'):
        S = special.betainc(k + 1.0, th, mu / (th + mu)); out = np.log(S)
    for i in np.flatnonzero(~(S > TAIL_SWITCH) & np.isfinite(mu) & (mu > 0)):
        out[i] = _nb_series(k[i], mu[i], th)
    return out.reshape(shape)


# ------------------------------------------------------------------ objectives (average NLL)
class Problem:
    def __init__(self, X, K, I, offset=None):
        self.X = np.asarray(X, float); self.K = np.asarray(K, float); self.I = np.asarray(I, bool)
        self.n, self.p = self.X.shape
        self.off = np.zeros(self.n) if offset is None else np.asarray(offset, float)
        sd = self.X.std(0); sd[sd == 0] = 1.0
        self.sd = sd                                  # standardized-scale gradient = g * sd

    def eta(self, b):
        return self.X @ b + self.off


def m_nll(b, P):
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        t = P.eta(b); mu = np.exp(t)
        ll = np.where(P.I, 0.0, P.K * t - mu - special.gammaln(P.K + 1.0))
        if P.I.any():
            ll[P.I] = pois_logsf(P.K[P.I], mu[P.I])
    return -np.mean(ll)


def m_hazard(b, P):
    """h = mu f(K)/S(K) on censored rows (log domain)."""
    t = P.eta(b)[P.I]; mu = np.exp(t); k = P.K[P.I]
    return np.exp(t + pois_logpmf(k, mu) - pois_logsf(k, mu)), mu, k


def m_grad(b, P):
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        mu = np.exp(P.eta(b)); s = P.K - mu
        if P.I.any():
            h, _, _ = m_hazard(b, P); s = s.copy(); s[P.I] = h
    return -(P.X.T @ s) / P.n


def _curv_series(k, mu, tol=1e-17):
    t, tm1, j = 1.0, 0.0, 2
    while True:
        t *= mu / (k + j); tm1 += t; j += 1
        if t < tol * (1.0 + tm1):
            return mu - (k + 1.0) * (tm1 / (1.0 + tm1))


def m_hess(b, P):
    """Hessian of the average NLL (positive definite at an interior optimum)."""
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        mu = np.exp(P.eta(b)); w = mu.copy()
        if P.I.any():
            h, muc, k = m_hazard(b, P)
            br = muc + h - k - 1.0
            for i in np.flatnonzero(k + 1.0 > 2.0 * muc):
                br[i] = _curv_series(k[i], muc[i])
            w[P.I] = h * br
    return (P.X.T * w) @ P.X / P.n


def nb_nll(th_, P):
    b, lth = th_[:-1], th_[-1]
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        th = math.exp(lth); mu = np.exp(P.eta(b))
        ll = np.where(P.I, 0.0, _nb_logpmf(P.K, mu, th))
        if P.I.any():
            ll[P.I] = nb_logsf(P.K[P.I], mu[P.I], th)
    return -np.mean(ll)


def nb_grad(th_, P, h=1e-6):
    b, lth = th_[:-1], th_[-1]
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        th = math.exp(lth); mu = np.exp(P.eta(b)); k = P.K
        s = k - mu * (k + th) / (th + mu)             # d log f / d eta (uncensored)
        if P.I.any():
            kc, muc = k[P.I], mu[P.I]
            # d log S(K)/d eta = mu * dS/dmu / S ;  dS/dmu = f_{th+1}(K; mu') * ... use identity
            # dS/dmu = (th/(th+mu)) * f(K)  * (K+th)/ (th+mu) * ... -> derived as: dS/dmu = f(K) * (K+th)/(th+mu)
            # (S(K) = I_q(K+1, th), q = mu/(th+mu); dI_q/dq = q^K (1-q)^(th-1)/B(K+1,th); dq/dmu = th/(th+mu)^2)
            lf = _nb_logpmf(kc, muc, th)
            ls = nb_logsf(kc, muc, th)
            s = s.copy()
            s[P.I] = muc * (kc + th) / (th + muc) * np.exp(lf - ls)
    gb = -(P.X.T @ s) / P.n
    e = np.zeros_like(th_); e[-1] = h
    gt = (nb_nll(th_ + e, P) - nb_nll(th_ - e, P)) / (2 * h)
    return np.concatenate([gb, [gt]])


def zip_nll(th_, P):
    b, t = th_[:-1], th_[-1]
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        lpi = -np.logaddexp(0.0, -t); l1pi = -np.logaddexp(0.0, t)
        mu = np.exp(P.eta(b)); k = P.K
        z0 = (~P.I) & (k == 0); up = (~P.I) & (k > 0)
        ll = np.empty(P.n)
        ll[z0] = np.logaddexp(lpi, l1pi - mu[z0])
        ll[up] = l1pi + pois_logpmf(k[up], mu[up])
        if P.I.any():
            ll[P.I] = l1pi + pois_logsf(k[P.I], mu[P.I])
    return -np.mean(ll)


def zip_grad(th_, P):
    b, t = th_[:-1], th_[-1]
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        lpi = -np.logaddexp(0.0, -t); l1pi = -np.logaddexp(0.0, t); pi = math.exp(lpi)
        tt = P.eta(b); mu = np.exp(tt); k = P.K
        z0 = (~P.I) & (k == 0); up = (~P.I) & (k > 0)
        sb = np.empty(P.n); st = np.empty(P.n)
        lden = np.logaddexp(lpi, l1pi - mu[z0])
        w0 = np.exp(l1pi - mu[z0] - lden)              # posterior share of the Poisson zero
        sb[z0] = -mu[z0] * w0
        st[z0] = (1 - w0) - pi                          # d/dt log(pi + (1-pi)e^-mu)
        sb[up] = k[up] - mu[up]; st[up] = -pi
        if P.I.any():
            sb[P.I] = np.exp(tt[P.I] + pois_logpmf(k[P.I], mu[P.I]) - pois_logsf(k[P.I], mu[P.I]))
            st[P.I] = -pi
    return -np.concatenate([P.X.T @ sb, [st.sum()]]) / P.n


# ------------------------------------------------------------------ finite-maximum screen (p columns)
def _rank(A):
    if A.shape[0] == 0:
        return 0
    sv = np.linalg.svd(A, compute_uv=False)
    return int((sv > max(A.shape) * np.finfo(float).eps * sv[0]).sum())


def screen(X, K, I):
    """'pass' | 'nonexistent' | 'unresolved' | 'unidentified' (rank(X) < p: not a recession question)."""
    p = X.shape[1]
    if _rank(np.unique(X, axis=0)) < p:
        return 'unidentified', float('nan')
    Up = np.unique(X[(~I) & (K > 0)], axis=0); U0 = np.unique(X[(~I) & (K == 0)], axis=0); C1 = np.unique(X[I], axis=0)
    if _rank(Up) == p:
        return 'pass', 0.0
    A_ub = np.vstack([U0, -C1]) if len(U0) + len(C1) else None
    r = optimize.linprog(-(C1.sum(0) - U0.sum(0)), A_ub=A_ub, b_ub=None if A_ub is None else np.zeros(len(A_ub)),
                         A_eq=Up if len(Up) else None, b_eq=np.zeros(len(Up)) if len(Up) else None,
                         bounds=[(-1, 1)] * p, method='highs')
    if r.status != 0:
        return 'unresolved', float('nan')
    v = r.x
    res = max([0.0] + list(np.abs(Up @ v)) + list(np.maximum(U0 @ v, 0)) + list(np.maximum(-(C1 @ v), 0)))
    if res > 1e-9:
        return 'unresolved', float('nan')
    J = -r.fun
    return ('nonexistent', J) if J > 1e-8 else ('pass', J) if J >= -1e-8 else ('unresolved', J)


# ------------------------------------------------------------------ fitting
def _minimize(fun, jac, x0, bounds, maxiter, P, gscale):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = optimize.minimize(fun, x0, args=(P,), jac=jac, method='L-BFGS-B', bounds=bounds,
                              options=dict(maxiter=maxiter, maxfun=10 * maxiter, maxls=50, gtol=SOLVER_GTOL, ftol=0.0))
    f = fun(r.x, P); g = jac(r.x, P)
    finite = bool(np.all(np.isfinite(r.x)) and np.isfinite(f) and np.all(np.isfinite(g)))
    # projected gradient: zero the component pushing against an active bound
    gp = g.copy()
    if bounds is not None:
        for j, (lo, hi) in enumerate(bounds):
            if lo is not None and r.x[j] <= lo + 1e-10 and g[j] > 0: gp[j] = 0.0
            if hi is not None and r.x[j] >= hi - 1e-10 and g[j] < 0: gp[j] = 0.0
    g_orig = float(np.max(np.abs(gp))) if finite else float('inf')
    g_std = float(np.max(np.abs(gp * gscale))) if finite else float('inf')
    at_bound = bool(bounds is not None and any((lo is not None and r.x[j] <= lo + 1e-10) or (hi is not None and r.x[j] >= hi - 1e-10)
                                               for j, (lo, hi) in enumerate(bounds)))
    return dict(x=r.x, obj=float(f), g_orig=g_orig, g_std=g_std, nit=int(r.nit), nfev=int(r.nfev), success=bool(r.success),
                msg=str(r.message), finite=finite, at_bound=at_bound,
                converged=bool(r.success and finite and g_orig <= GTOL and g_std <= GTOL))


def fit(model, X, K, I, starts, offset=None, maxiter=None):
    """model in {'M','NB2','MIX'}; starts = list of initial vectors (beta, or beta+[aux]).
    Returns (fit_record, start_records). Best = lowest objective among converged starts."""
    P = Problem(X, K, I, offset)
    scr, J = screen(P.X, P.K, P.I)
    if model == 'M':
        fun, jac, bounds, gs, mi = m_nll, m_grad, None, P.sd, maxiter or 5000
    elif model == 'NB2':
        fun, jac, bounds, gs, mi = nb_nll, nb_grad, [(None, None)] * P.p + [THETA_BOUNDS], np.append(P.sd, 1.0), maxiter or 5000
    elif model == 'MIX':
        fun, jac, bounds, gs, mi = zip_nll, zip_grad, [(None, None)] * P.p + [PI_LOGIT_BOUNDS], np.append(P.sd, 1.0), maxiter or 5000
    else:
        raise ValueError(model)
    recs, best = [], None
    for j, x0 in enumerate(starts):
        if scr != 'pass':
            recs.append(dict(start=j, attempted=False, skip_reason=f'screen_{scr}')); continue
        o = _minimize(fun, jac, np.asarray(x0, float), bounds, mi, P, gs)
        recs.append(dict(start=j, attempted=True, skip_reason='', success=o['success'], msg=o['msg'], objective=o['obj'],
                         g_orig=o['g_orig'], g_std=o['g_std'], nit=o['nit'], nfev=o['nfev'], at_bound=o['at_bound'],
                         converged=o['converged']))
        if o['converged'] and (best is None or o['obj'] < best[1]['obj'] - 1e-12):
            best = (j, o)
    out = dict(model=model, screen=scr, screen_J=J, n=P.n, p=P.p, n_cens=int(P.I.sum()))
    if scr == 'unidentified':
        out['status'] = 'unidentified'
    elif scr in ('nonexistent', 'unresolved'):
        out['status'] = {'nonexistent': 'structural_nonexistence', 'unresolved': 'unresolved_screen'}[scr]
    elif best is None:
        att = [r for r in recs if r.get('attempted')]
        out['status'] = 'nonfinite' if all(not np.isfinite(r['objective']) for r in att) else 'nonconvergence'
    else:
        j, o = best
        out.update(status='valid', selected_start=j, x=o['x'], objective=o['obj'], g_orig=o['g_orig'], g_std=o['g_std'],
                   nit=o['nit'], at_bound=o['at_bound'])
        e = P.eta(o['x'][:P.p]); out.update(eta_min=float(e.min()), eta_max=float(e.max()))
        if model == 'M':
            H = m_hess(o['x'], P)
            if np.all(np.isfinite(H)):
                ev = np.linalg.eigvalsh(H); tau = 1e-10 * max(1.0, abs(ev[-1]))
                out.update(eig_min=float(ev[0]), eig_max=float(ev[-1]),
                           info='ok' if ev[0] > tau else ('singular_or_nearly' if ev[0] >= -tau else 'indefinite'),
                           cond=float(ev[-1] / ev[0]) if ev[0] > tau else float('nan'))
            else:
                out.update(info='missing')
    out['valid'] = out['status'] == 'valid'
    return out, recs
