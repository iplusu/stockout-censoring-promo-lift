"""Corrected drop-in for pilot/tobit.py (same public API: fit, fit_fe, init_poisson_irls, latent_day).
Used ONLY inside the P5 sandbox (sandbox/pilot/tobit.py) so every existing consumer script routes through the
corrected strict-tail censored Poisson (mfit_corrected.py, frozen solver contract, tag p2-frozen) without editing
the consumers' estimation logic. Legacy file is untouched (legacy/pilot/tobit.py).

Contract differences vs legacy, all intentional:
  fit       : multistart per solver_contract (b0, b0+0.1 e_promo, b0+0.1 e_depth when p >= 3); `converged` = contract
              validity (stationarity on original and standardized scale), NOT res.success. Non-valid fits return
              b = NaN (no arbitrary stopping point leaks downstream) plus `status`.
  latent_day: E[D | D > K] = lambda + h with the stable strict tail; no lambda clipping or tail floor.
  fit_fe    : joint Newton MLE of identified slopes and group offsets (p4_hourly_f.fit_f); absorbed columns -> NaN;
              offsets of separated groups are +/-inf, so recovery there is NaN (consumers must count, not impute).
"""
import os, sys
import numpy as np

CORR = os.environ.get('JAMS_CORR_DIR', '/home/ubuntu/papers/jams-frn50k/correction_20261122')
if CORR not in sys.path:
    sys.path.insert(0, CORR)
import mfit_corrected as mf  # noqa: E402

_LEG = os.path.join(CORR, 'legacy', 'pilot')


def _legacy_init():
    import importlib.util
    spec = importlib.util.spec_from_file_location('tobit_legacy', os.path.join(_LEG, 'tobit.py'))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.init_poisson_irls


init_poisson_irls = _legacy_init()     # same start values as legacy (solver contract)


_CACHE = os.path.join(CORR, 'out', 'shim_cache')


def fit(X, S, c, b0=None, offset=None, maxiter=500):
    """Disk-memoized on the exact inputs (deterministic fit), so consumer scripts refitting the same cell reuse it."""
    import hashlib, pickle
    h = hashlib.sha256()
    for a in (X, S, c, b0, offset):
        h.update(b'|' if a is None else np.ascontiguousarray(np.asarray(a, float)).tobytes() + str(np.shape(a)).encode())
    path = os.path.join(_CACHE, h.hexdigest()[:32] + '.pkl')
    if os.path.exists(path):
        return pickle.load(open(path, 'rb'))
    r = _fit(X, S, c, b0, offset)
    os.makedirs(_CACHE, exist_ok=True); tmp = path + f'.{os.getpid()}'
    pickle.dump(r, open(tmp, 'wb')); os.replace(tmp, path)
    return r


def _fit(X, S, c, b0=None, offset=None):
    X = np.asarray(X, float); K = np.asarray(S, float); I = np.asarray(c).astype(bool); p = X.shape[1]
    if b0 is None:
        b0 = init_poisson_irls(X, S, c)
    b0 = np.asarray(b0, float)
    starts = [b0]
    for j in (1, 2):
        if p > j:
            e = np.zeros(p); e[j] = 0.1; starts.append(b0 + e)
    out, recs = mf.fit('M', X, K, I, starts, offset=offset)
    valid = out['valid']
    b = np.asarray(out['x'], float) if valid else np.full(p, np.nan)
    se = None
    ll = float('nan')
    gnorm = float('nan')
    if valid:
        P = mf.Problem(X, K, I, offset)
        ll = -mf.m_nll(b, P) * P.n
        gnorm = float(np.linalg.norm(mf.m_grad(b, P) * P.n))
        try:
            se = np.sqrt(np.diag(np.linalg.inv(mf.m_hess(b, P) * P.n)))
            if not np.all(np.isfinite(se)):
                se = None
        except np.linalg.LinAlgError:
            se = None
    att = [r for r in recs if r.get('attempted')]
    return dict(b=b, ll=float(ll), success=valid, nit=int(out.get('nit', max([r.get('nit', 0) for r in att], default=0))),
                msg=out['status'], converged=valid, grad_norm=gnorm, b0=b0.copy(), se=se, status=out['status'],
                g_orig=out.get('g_orig'), g_std=out.get('g_std'), info=out.get('info'))


def latent_day(b, X, S, c, offset=None):
    X = np.asarray(X, float); K = np.asarray(S, float); I = np.asarray(c).astype(bool)
    b = np.asarray(b, float)
    eta = X @ b + (0.0 if offset is None else offset)
    lam = np.exp(eta)
    D = K.copy()
    if not np.all(np.isfinite(b)):
        return np.full(len(K), np.nan), lam
    if I.any():
        P = mf.Problem(np.zeros((len(K), 1)), K, I, eta)
        with np.errstate(all='ignore'):
            h, mu, _ = mf.m_hazard(np.zeros(1), P)
        D[I] = mu + h
    return D, lam


def fit_fe(X, S, c, keys, max_iter=25, maxiter_mle=200, verbose=False, b0=None, alpha0=None):
    import p4_hourly_f as p4
    X = np.asarray(X, float)
    uniq, inv = np.unique(keys, return_inverse=True)
    bs = b0 if b0 is not None else init_poisson_irls(X, S, c)
    rec, _, beta_full, alpha_full = p4.fit_f(X, np.asarray(S), np.asarray(c), inv, bs, return_params=True)
    valid = rec['status'] == 'valid'
    return dict(b=beta_full if valid else np.full(X.shape[1], np.nan), alpha=alpha_full, n_groups=int(len(uniq)),
                converged=valid, success=valid, nit=int(rec.get('nit', 0)), ll=float('nan'),  # objective omits lgamma(K+1); not comparable to legacy ll
               
                grad_norm=float('nan') if not valid else rec['final']['g_beta'], status=rec['status'], fe_record=rec)
