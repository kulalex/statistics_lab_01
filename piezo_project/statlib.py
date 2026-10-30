"""Статистики, подгонка распределений и графики для лабораторной работы."""
import numpy as np, pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

RNG_SEED = 20261030
C = {'data': '#4C78A8', 'fit': '#E45756', 'alt': '#54A24B', 'grey': '#7f7f7f'}


def set_style():
    plt.rcParams.update({'figure.dpi': 110, 'axes.grid': True, 'grid.alpha': .25, 'axes.spines.top': False,
                         'axes.spines.right': False, 'font.size': 10, 'axes.titlesize': 11,
                         'axes.titleweight': 'bold', 'figure.autolayout': True})

def summary(x):
    x = np.asarray(x, float)
    q = np.percentile(x, [25, 50, 75])
    iqr = q[2] - q[0]
    return pd.Series({
        'n': len(x), 'mean': x.mean(), 'std (ddof=1)': x.std(ddof=1), 'min': x.min(), 'Q1': q[0],
        'median': q[1], 'Q3': q[2], 'max': x.max(), 'IQR': iqr, 'skew': stats.skew(x),
        'ex.kurtosis': stats.kurtosis(x), 'CV': x.std(ddof=1) / abs(x.mean()),
        'выбросы (1.5·IQR)': int(((x < q[0] - 1.5 * iqr) | (x > q[2] + 1.5 * iqr)).sum())})


def determinism_check(df, target, feats, seed=RNG_SEED):
    """Проверка предсказания по доступным дескрипторам; не тест физического детерминизма."""
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_predict, GroupKFold
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.metrics import r2_score
    d = df.dropna(subset=feats + [target])
    X, y = d[feats], d[target].values
    groups = d.formula.to_numpy()
    kf = GroupKFold(5)
    cat = [f for f in feats if f == 'system_code']
    numeric = [f for f in feats if f not in cat]
    def prep():
        return ColumnTransformer([('num', StandardScaler(), numeric),
                                  ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat)])
    out = {}
    for name, mdl in (('Ridge', make_pipeline(prep(), RidgeCV(alphas=np.logspace(-3, 3, 13)))),
                      ('GradBoost', make_pipeline(prep(), GradientBoostingRegressor(random_state=seed)))):
        p = cross_val_predict(mdl, X, y, cv=kf, groups=groups)
        res = y - p
        out[name] = {'CV R²': r2_score(y, p), 'N (CV)': len(d), 'RMSE': np.sqrt((res ** 2).mean()),
                     'SD(y)': y.std(ddof=1), 'остатки: SD': res.std(ddof=1)}
    return pd.DataFrame(out).T


def formula_spread(df, target, min_group=3):
    """Разброс свойства среди структур с одинаковой формулой."""
    g = df.groupby('formula')[target].agg(['count', 'mean', 'std', 'min', 'max'])
    g = g[g['count'] >= min_group]
    g['range'] = g['max'] - g['min']
    return g.sort_values('count', ascending=False)

CANDS = {n: getattr(stats, n) for n in ['norm', 'lognorm', 'gamma', 'weibull_min', 'invgauss', 'fisk', 'lomax', 'genpareto',
                                         'burr12', 'genextreme', 'johnsonsu', 'skewnorm', 'invgamma', 'expon']}


def fit_table(x, names=None, floc=None):
    """MLE-подгонка кандидатов; floc — зафиксированный параметр положения (напр. 0 для положительных величин)."""
    x = np.asarray(x, float)
    rows = []
    for nm in (names or CANDS):
        d = CANDS[nm]
        try:
            p = d.fit(x, floc=floc) if floc is not None else d.fit(x)
            k = len(p) - (1 if floc is not None else 0)
            ll = d.logpdf(x, *p).sum()
            if not np.isfinite(ll):
                raise ValueError(f'{nm}: неконечное правдоподобие')
            ks = stats.kstest(x, d.cdf, args=p).statistic
            rows.append({'dist': nm, 'k': k, 'logL': ll, 'AIC': 2 * k - 2 * ll, 'BIC': k * np.log(len(x)) - 2 * ll,
                         'KS D': ks, 'params': tuple(np.round(p, 4))})
        except Exception as exc:
            import warnings
            warnings.warn(f'Семейство {nm} исключено: {exc}', RuntimeWarning)
    t = pd.DataFrame(rows).sort_values('AIC').reset_index(drop=True)
    t['ΔAIC'] = t.AIC - t.AIC.min()
    return t


def param_bootstrap_ks(dist, x, B=300, seed=RNG_SEED, floc=None):
    """КС с оценёнными параметрами (идея Лиллиефорса): p-value параметрическим бутстрепом."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    fit = (lambda s: dist.fit(s, floc=floc)) if floc is not None else dist.fit
    p = fit(x)
    d0 = stats.kstest(x, dist.cdf, args=p).statistic
    cnt = 0
    for _ in range(B):
        xb = dist.rvs(*p, size=len(x), random_state=rng)
        pb = fit(xb)
        cnt += stats.kstest(xb, dist.cdf, args=pb).statistic >= d0
    return p, d0, (cnt + 1) / (B + 1)

def consistency_curve(pop, estimators, ns, reps=500, seed=RNG_SEED, replace=False):
    """Для каждого n: reps выборок из 'ген. совокупности', ошибка оценки |θ̂−θ|, смещение и SD."""
    rng = np.random.default_rng(seed)
    rows = []
    truth = {k: (tf or f)(pop) for k, (f, tf) in estimators.items()}   # tf — функция оцениваемого параметра
    for n in ns:
        est = {k: [] for k in estimators}
        for _ in range(reps):
            s = rng.choice(pop, n, replace=replace)
            for k, (f, _) in estimators.items():
                est[k].append(f(s))
        for k in estimators:
            e = np.array(est[k])
            rows.append({'n': n, 'estimator': k, 'truth': truth[k], 'mean_est': e.mean(),
                         'bias': e.mean() - truth[k], 'sd': e.std(ddof=1),
                         'MAE': np.abs(e - truth[k]).mean(),
                         'eps': 0.1 * abs(truth[k]),
                         'P(|err|>eps)': np.mean(np.abs(e - truth[k]) > 0.1 * abs(truth[k]))})
    return pd.DataFrame(rows)

def sampling_experiment(pop, n, reps=5000, replace=True, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    means = np.empty(reps); vars_ = np.empty(reps); meds = np.empty(reps)
    for i in range(reps):
        s = rng.choice(pop, n, replace=replace)
        means[i], vars_[i], meds[i] = s.mean(), s.var(ddof=1), np.median(s)
    N = len(pop)
    sigma2 = pop.var(ddof=0)            # дисперсия ген. совокупности (деление на N)
    S2 = pop.var(ddof=1)                # исправленная
    if replace:
        se_theory = np.sqrt(sigma2 / n)
    else:
        se_theory = np.sqrt(S2 / n * (1 - n / N))   # с поправкой на конечную совокупность
    return {'n': n, 'replace': replace, 'E[x̄]': means.mean(), 'SE(x̄) эмпир.': means.std(ddof=1),
            'SE(x̄) теор.': se_theory, 'E[s²]': vars_.mean(), 'σ² (N)': sigma2, 'S² (N−1)': S2,
            'SE(median)': meds.std(ddof=1)}


def one_sample_report(pop, n, replace, seed=RNG_SEED):
    rng = np.random.default_rng(seed + n + int(replace))
    s = rng.choice(pop, n, replace=replace)
    ks = stats.ks_2samp(s, pop)
    return {'n': n, 'replace': replace, 'mean': s.mean(), 'std': s.std(ddof=1), 'median': np.median(s),
            'skew': stats.skew(s), 'kurt': stats.kurtosis(s), 'KS D (описательно)': ks.statistic}

def independence_report(a, b, bins=3, B=5000, seed=RNG_SEED):
    a, b = np.asarray(a, float), np.asarray(b, float)
    pr, spr, kt = stats.pearsonr(a, b), stats.spearmanr(a, b), stats.kendalltau(a, b)
    qa = pd.qcut(a, bins, labels=False, duplicates='drop'); qb = pd.qcut(b, bins, labels=False, duplicates='drop')
    ct = pd.crosstab(qa, qb)
    chi2, p_chi, dof, _ = stats.chi2_contingency(ct)
    cramer = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
    from sklearn.feature_selection import mutual_info_regression
    mi = mutual_info_regression(a.reshape(-1, 1), b, random_state=seed)[0]
    rng = np.random.default_rng(seed)
    cnt = sum(abs(stats.spearmanr(a, rng.permutation(b))[0]) >= abs(spr[0]) for _ in range(B))
    return {'n': len(a), 'Pearson r': pr[0], 'p (Pearson, норм. допущ.)': pr[1], 'Spearman ρ': spr[0], 'p (Spearman)': spr[1],
            "Kendall τ": kt[0], 'p (Kendall)': kt[1], f'χ² ({bins}×{bins})': chi2, 'p (χ²)': p_chi,
            "Cramér V": cramer, 'MI (нат)': mi, 'p (перест., |ρ|)': (cnt + 1) / (B + 1)}, ct


# графики
def eda_figure(x, label, dist=None, params=None, dist_name=''):
    """6 панелей для положительных величин с тяжёлым хвостом."""
    x = np.sort(np.asarray(x, float)); n = len(x)
    p = (np.arange(1, n + 1) - .5) / n
    fig, ax = plt.subplots(2, 3, figsize=(14, 7.5))
    q99 = np.quantile(x, .99)
    a = ax[0, 0]
    edges = np.histogram_bin_edges(x[x <= q99], bins='fd')
    counts, _ = np.histogram(x, bins=edges)
    a.bar(edges[:-1], counts / (n * np.diff(edges)), width=np.diff(edges), align='edge',
          color=C['data'], alpha=.55, label='данные (до 99-го перцентиля)')
    g = np.linspace(x.min(), q99, 400)
    if dist is not None:
        a.plot(g, dist.pdf(g, *params), color=C['fit'], lw=2, label=f'{dist_name}')
    a.set(xlabel=label, ylabel='плотность', title='Гистограмма (хвост обрезан)'); a.legend(fontsize=7)
    a = ax[0, 1]
    lx = np.log10(x); a.hist(lx, bins='fd', density=True, color=C['data'], alpha=.55)
    gg = np.linspace(lx.min(), lx.max(), 400)
    if dist is not None:
        a.plot(gg, dist.pdf(10 ** gg, *params) * 10 ** gg * np.log(10), color=C['fit'], lw=2)
    a.set(xlabel=f'lg({label})', ylabel='плотность', title='Гистограмма в лог-шкале')
    a = ax[0, 2]
    a.boxplot(x, vert=False, widths=.5, flierprops=dict(marker='.', markersize=3, alpha=.4)); a.set_xscale('log')
    a.set(xlabel=label, yticks=[], title='Ящик с усами (лог-шкала)')
    a = ax[1, 0]
    d = dist if dist is not None else stats.norm
    pr = params if dist is not None else (x.mean(), x.std(ddof=1))
    q = d.ppf(p, *pr); ok = np.isfinite(q)
    a.plot(q[ok], x[ok], '.', ms=4, color=C['data'], alpha=.6)
    lo, hi = min(q[ok].min(), x[ok].min()), max(q[ok].max(), x[ok].max())
    a.plot([lo, hi], [lo, hi], color=C['fit'])
    a.set(xlabel=f'теор. квантили ({dist_name or "норм."})', ylabel='выборочные квантили', title='Q–Q')
    a = ax[1, 1]
    a.step(x, np.arange(1, n + 1) / n, where='post', color=C['data'], label='ЭФР'); gl = np.logspace(np.log10(x.min()), np.log10(x.max()), 400)
    a.plot(gl, d.cdf(gl, *pr), color=C['fit'], label='ФР модели'); a.set_xscale('log')
    a.set(xlabel=label, ylabel='F(x)', title='ЭФР и ФР модели'); a.legend(fontsize=7)
    a = ax[1, 2]
    ux, freq = np.unique(x, return_counts=True)
    survival = 1 - np.cumsum(freq) / n
    keep = survival > 0
    a.loglog(ux[keep], survival[keep], '.', ms=4, color=C['data'], label='1 − ЭФР')
    a.loglog(gl, d.sf(gl, *pr), color=C['fit'], label='1 − ФР модели')
    a.set(xlabel=label, ylabel='P(X > x)', title='Хвост распределения (лог-лог)'); a.legend(fontsize=7)
    return fig


def normalization_table(x):
    from sklearn.preprocessing import PowerTransformer, QuantileTransformer
    x = np.asarray(x, float).reshape(-1, 1)
    outs = {'исходная': x.ravel(), 'z-score': ((x - x.mean()) / x.std(ddof=1)).ravel(),
            'min–max': ((x - x.min()) / (x.max() - x.min())).ravel(),
            'Yeo–Johnson': PowerTransformer('yeo-johnson').fit_transform(x).ravel(),
            'квантиль→N(0,1)': QuantileTransformer(n_quantiles=min(1000, len(x)), output_distribution='normal',
                                                  random_state=RNG_SEED).fit_transform(x).ravel()}
    rows = []
    for k, v in outs.items():
        rows.append({'преобразование': k, 'mean': v.mean(), 'std': v.std(ddof=1), 'min': v.min(), 'max': v.max(),
                     'skew': stats.skew(v), 'ex.kurt': stats.kurtosis(v), 'p K² (после подгонки — описательно)': stats.normaltest(v).pvalue})
    return pd.DataFrame(rows).set_index('преобразование'), outs


def consistency_figure(tab, est_names, label):
    fig, axes = plt.subplots(len(est_names), 2, figsize=(11, 3.1 * len(est_names)), squeeze=False)
    for i, key in enumerate(est_names):
        t = tab[tab.estimator == key]
        axes[i, 0].plot(t.n, t.sd, 'o-', color=C['data'])
        axes[i, 0].set(xscale='log', xlabel='объём выборки', ylabel='SD оценки', title=key)
        axes[i, 1].plot(t.n, t.bias, 'o-', color=C['fit'])
        axes[i, 1].axhline(0, color='k', lw=1)
        axes[i, 1].set(xscale='log', xlabel='объём выборки', ylabel='смещение оценки', title=key)
    fig.suptitle(f'{label}: характеристики оценок при отборе из конечной базы', fontsize=11)
    return fig


def holm_adjust(pvalues):
    """Поправка Холма, сохраняющая порядок исходных гипотез."""
    p = np.asarray(pvalues, dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('p-value должны быть конечными числами от 0 до 1')
    order = np.argsort(p)
    adjusted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    result = np.empty_like(p)
    result[order] = np.minimum(adjusted, 1)
    return result
