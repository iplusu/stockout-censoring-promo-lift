"""검열 카운트 MLE — Tobit형 가능도 (Poisson 잠복 수요).
일 단위: D ~ Poisson(lambda), lambda = exp(X b + offset).
관찰: 스톡아웃 없으면 S = D (정확), 있으면 D > S (판매는 하한).
ll_i = c_i*log(1-F(S;lam)) + (1-c_i)*(-lam + S log lam - ln S!)
잠복 환원: E[D|D>S] = lambda + lambda*f(S)/Pbar(S);  uncensored: S.
(연속 Tobit은 판매의 0.1격자/0점질량에서 치명적 bias — 합성 데이터로 확인 후 Poisson 교체)
"""
import numpy as np
from scipy import optimize, special


def _logf_pois(s, lam):
    return -lam + s * np.log(lam) - special.gammaln(s + 1.0)


def _logpbar(s, lam):
    # log P(D > s) = log gammainc(s+1, lam)  (정규화 하한 감마 = Poisson 상단 꼬리)
    up = special.gammainc(s + 1.0, lam)
    return np.log(np.clip(up, 1e-300, 1.0))


def _ll_grad(b, X, S, c, offset=None, lam_clip=(1e-8, 1e6)):
    eta = X @ b + (0.0 if offset is None else offset)
    lam = np.clip(np.exp(eta), *lam_clip)
    s = S.astype(np.float64)
    lpf = _logf_pois(s, lam)
    lpb = _logpbar(s, lam)
    ratio = np.exp(np.clip(lpf - lpb, -700, 700))
    ll = float(np.sum(c * lpb + (1.0 - c) * (-lam + s * np.log(lam) - special.gammaln(s + 1.0))))
    # censored: d log P(D>S)/d eta = (f/Pbar)*lambda  (dF/dlambda = -f);  uncensored: S - lambda
    sc = c * lam * ratio + (1.0 - c) * (s - lam)
    g = X.T @ sc
    return ll, g


def fit(X, S, c, b0=None, offset=None, maxiter=500):
    n, k = X.shape
    if b0 is None:
        b0 = np.zeros(k)
        uc = c == 0
        b0[0] = np.log(max(S[uc].mean() + 0.3, 1e-3))
    t = b0.astype(float).copy()

    def negll(th):
        ll, g = _ll_grad(th, X, S, c, offset)
        return -ll, -g

    res = optimize.minimize(negll, t, jac=True, method='L-BFGS-B',
                            options=dict(maxiter=maxiter, ftol=1e-12, gtol=1e-8))
    ll, g = _ll_grad(res.x, X, S, c, offset)
    converged = bool(res.success and np.isfinite(ll))
    # ponytail: approximate Wald SE via the L-BFGS-B inverse-Hessian product. res.hess_inv is an
    # LbfgsInvHessProduct (no .toarray); k is tiny so reconstruct k x k by mat-vec on identity.
    # E2-v2 CI-coverage diagnostic only -- never used for a manuscript point estimate.
    se = None
    if converged:
        try:
            kth = res.x.shape[0]
            se = np.sqrt(np.clip(np.diag(res.hess_inv.dot(np.eye(kth))), 0.0, None))
            if not np.all(np.isfinite(se)):
                se = None
        except Exception:
            se = None
    return dict(b=res.x, ll=float(ll), success=res.success, nit=int(res.nit),
                msg=str(res.message), converged=converged,
                grad_norm=float(np.linalg.norm(g)), b0=b0.copy(), se=se)


def init_poisson_irls(X, S, c, n_iter=10):
    """uncensored만 Poisson GLM IRLS — MLE 초기값용."""
    uc = c == 0
    Ss, Xs = S[uc], X[uc]
    b = np.zeros(X.shape[1])
    b[0] = np.log(max(Ss.mean() + 0.3, 1e-3))
    for _ in range(n_iter):
        eta = np.clip(Xs @ b, -30, 30)
        lam = np.exp(eta)
        w = lam
        z = eta + (Ss - lam) / np.maximum(lam, 1e-8)
        XtW = Xs.T @ (w[:, None] * Xs)
        try:
            b = np.linalg.solve(XtW + 1e-12 * np.eye(Xs.shape[1]), Xs.T @ (w * z))
        except np.linalg.LinAlgError:
            break
    return b


def latent_day(b, X, S, c, offset=None):
    """E[D | data] (일 단위, count). censored: lambda + lambda*f(S)/Pbar(S)."""
    eta = X @ b + (0.0 if offset is None else offset)
    lam = np.clip(np.exp(eta), 1e-8, 1e6)
    s = S.astype(np.float64)
    ratio = np.exp(np.clip(_logf_pois(s, lam) - _logpbar(s, lam), -700, 700))
    D = np.where(c == 0, s, lam + lam * ratio)
    return np.maximum(D, 0.0), lam


def _scores_at(b, X, S, c, offset):
    eta = X @ b + offset
    lam = np.clip(np.exp(eta), 1e-8, 1e6)
    s = S.astype(np.float64)
    ratio = np.exp(np.clip(_logf_pois(s, lam) - _logpbar(s, lam), -700, 700))
    sc = c * lam * ratio + (1.0 - c) * (s - lam)
    # d2 ll / d alpha^2 (per obs): uncensored -lam; censored -(s*lam*r - lam^2*r^2)
    hess = np.where(c == 0, -lam, -(s * lam * ratio - (lam * ratio) ** 2))
    return sc, hess, lam


def fit_fe(X, S, c, keys, max_iter=25, maxiter_mle=200, verbose=False, b0=None, alpha0=None):
    """store-day FE 교대 MLE: offset으로 (b MLE | alpha) <-> (alpha Gauss-Newton | b), 발산 시 step halving.
    b0/alpha0 (E5 멀티스타트용): 기본 None = 기존 동작 불변."""
    uniq, inv = np.unique(keys, return_inverse=True)
    n_g = len(uniq)
    alpha = np.zeros(n_g) if alpha0 is None else np.asarray(alpha0, dtype=float).copy()
    b = b0
    last_ll = -np.inf
    for it in range(max_iter):
        off = alpha[inv]
        f = fit(X, S, c, b0=b, offset=off, maxiter=maxiter_mle)
        b = f['b']
        sc, hess, lam = _scores_at(b, X, S, c, off)
        step, ok, da = 1.0, False, None
        for _try in range(3):
            q = np.bincount(inv, weights=sc, minlength=n_g)
            h = np.bincount(inv, weights=-hess, minlength=n_g)
            da = np.divide(q, np.maximum(h, 1e-8), out=np.zeros(n_g), where=h > 1e-8)
            alpha_try = alpha + step * da
            ll_try, _ = _ll_grad(b, X, S, c, alpha_try[inv])
            if ll_try >= last_ll - 1e-9 or step < 0.125:
                alpha, last_ll = alpha_try, max(ll_try, last_ll)
                ok = True
                break
            step *= 0.5
        if verbose:
            print(f'  fe iter {it}: ll={f["ll"]:.1f} max|da|={np.max(np.abs(da)):.5f}')
        if ok and it > 0 and step * np.max(np.abs(da)) < 1e-8:
            break
    f = fit(X, S, c, b0=b, offset=alpha[inv], maxiter=maxiter_mle)
    f['alpha'] = alpha
    f['n_groups'] = int(n_g)
    return f
