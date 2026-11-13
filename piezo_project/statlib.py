"""Статистики, подгонка распределений и графики для лабораторной работы."""
import numpy as np, pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import labstyle as L

RNG_SEED = 20261030
C = L.C


def set_style():
    L.apply()

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
def axis_label(sym, unit):
    return f'{sym}, {unit}' if unit else f'{sym}, безразмерный'


def eda_figure(x, sym, unit='', dist=None, params=None, dist_name=''):
    """6 панелей для положительных величин с тяжёлым хвостом.

    sym — обозначение величины, unit — единица измерения (пустая строка для безразмерной).
    Если распределение не задано, на панелях (г)–(е) для сравнения берётся нормальное с выборочными средним и SD.
    """
    x = np.sort(np.asarray(x, float)); n = len(x)
    p = (np.arange(1, n + 1) - .5) / n
    label = axis_label(sym, unit)
    dens_unit = f', ({unit})⁻¹' if unit else ''
    model = dist_name or 'нормальное'
    fig, ax = plt.subplots(2, 3, figsize=(L.FULL, 2 * L.PANEL))
    q99 = np.quantile(x, .99)
    a = ax[0, 0]
    edges = np.histogram_bin_edges(x[x <= q99], bins='fd')
    counts, _ = np.histogram(x, bins=edges)
    a.bar(edges[:-1], counts / (n * np.diff(edges)), width=np.diff(edges), align='edge',
          color=C['data'], alpha=.6, label='данные до 99-го перцентиля')
    g = np.linspace(x.min(), q99, 400)
    if dist is not None:
        a.plot(g, dist.pdf(g, *params), color=C['model'], label=dist_name)
    a.set(xlabel=label, ylabel=f'плотность вероятности{dens_unit}'); a.legend(fontsize=8)
    L.letter(a, 0, 'Гистограмма, хвост обрезан')
    a = ax[0, 1]
    lx = np.log10(x); a.hist(lx, bins='fd', density=True, color=C['data'], alpha=.6, label='данные')
    gg = np.linspace(lx.min(), lx.max(), 400)
    if dist is not None:
        a.plot(gg, dist.pdf(10 ** gg, *params) * 10 ** gg * np.log(10), color=C['model'], label=dist_name)
    a.set(xlabel=f'$\\lg$ {sym}' + (f'  ({sym} в {unit})' if unit else ''), ylabel='плотность вероятности на декаду')
    L.letter(a, 1, 'Гистограмма логарифма')
    a = ax[0, 2]
    a.boxplot(x, orientation='horizontal', widths=.5, patch_artist=True,
              boxprops=dict(facecolor='white', edgecolor=C['data'], linewidth=1.6),
              medianprops=dict(color=C['ink'], linewidth=1.6), whiskerprops=dict(color=C['data'], linewidth=1.4),
              capprops=dict(color=C['data'], linewidth=1.4),
              flierprops=dict(marker='.', markersize=3, alpha=.4, markerfacecolor=C['data'], markeredgecolor=C['data']))
    a.set_xscale('log')
    a.set(xlabel=f'{label} (лог. шкала)', yticks=[]); L.plain_log(a)
    L.letter(a, 2, 'Ящик с усами')
    a = ax[1, 0]
    d = dist if dist is not None else stats.norm
    pr = params if dist is not None else (x.mean(), x.std(ddof=1))
    q = d.ppf(p, *pr); ok = np.isfinite(q)
    a.plot(q[ok], x[ok], '.', ms=4, color=C['data'], alpha=.6, label='данные')
    lo, hi = min(q[ok].min(), x[ok].min()), max(q[ok].max(), x[ok].max())
    a.plot([lo, hi], [lo, hi], color=C['model'], label='точное совпадение')
    a.set(xlabel=f'квантили модели: {model}' + (f', {unit}' if unit else ''),
          ylabel='квантили данных' + (f', {unit}' if unit else ''))
    a.legend(fontsize=8, loc='lower right')
    L.letter(a, 3, 'Q–Q график')
    a = ax[1, 1]
    a.step(x, np.arange(1, n + 1) / n, where='post', color=C['data'], label='данные (ЭФР)'); gl = np.logspace(np.log10(x.min()), np.log10(x.max()), 400)
    a.plot(gl, d.cdf(gl, *pr), color=C['model'], lw=1.6, label=f'модель: {model}'); a.set_xscale('log')
    a.set(xlabel=f'{label} (лог. шкала)', ylabel='$F(x)$'); a.legend(fontsize=8, loc='lower right'); L.plain_log(a)
    L.letter(a, 4, 'Функция распределения')
    a = ax[1, 2]
    ux, freq = np.unique(x, return_counts=True)
    survival = 1 - np.cumsum(freq) / n
    keep = survival > 0
    a.loglog(ux[keep], survival[keep], '.', ms=4, color=C['data'], label='данные')
    a.loglog(gl, d.sf(gl, *pr), color=C['model'], lw=1.6, label=f'модель: {model}')
    a.set(xlabel=f'{label} (лог. шкала)', ylabel='$P(X > x)$ (лог. шкала)', ylim=(0.3 / n, 1.5)); a.legend(fontsize=8, loc='lower left'); L.plain_log(a)
    L.letter(a, 5, 'Хвост распределения')
    return fig


def scatter_figure(x, y, xlabel, ylabel):
    """Диаграмма рассеяния в двойной логарифмической шкале."""
    fig, ax = plt.subplots(figsize=(L.HALF, L.HALF * .8))
    ax.loglog(x, y, '.', ms=5, color=C['data'], alpha=.5)
    ax.set(xlabel=f'{xlabel} (лог. шкала)', ylabel=f'{ylabel} (лог. шкала)'); L.plain_log(ax, 'x'); L.plain_log(ax, 'y')
    return fig


def parity_figure(y, pred, xlabel, ylabel):
    """Предсказание против расчётного значения; диагональ — точное совпадение."""
    fig, ax = plt.subplots(figsize=(L.HALF, L.HALF * .9))
    ax.loglog(y, pred, '.', ms=3, alpha=.4, color=C['data'], label='записи датасета')
    lo, hi = min(y.min(), pred.min()) * .9, max(y.max(), pred.max()) * 1.1
    ax.plot([lo, hi], [lo, hi], color=C['ref'], lw=1.4, ls='--', label='точное совпадение')
    ax.set(xlabel=f'{xlabel} (лог. шкала)', ylabel=f'{ylabel} (лог. шкала)', xlim=(lo, hi), ylim=(lo, hi))
    ax.set_aspect('equal'); ax.legend(loc='upper left'); L.plain_log(ax, 'x'); L.plain_log(ax, 'y')
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


def consistency_figure(tab, est_names, units):
    """Разброс (слева) и смещение (справа) оценок при росте объёма подвыборки. units: единица для каждой оценки."""
    fig, axes = plt.subplots(len(est_names), 2, figsize=(L.FULL, 2.7 * len(est_names)), squeeze=False)
    k = 0
    for i, key in enumerate(est_names):
        t = tab[tab.estimator == key]
        unit = units[key]
        for j, (col, what) in enumerate((('sd', 'SD оценки'), ('bias', 'смещение оценки'))):
            ax = axes[i, j]
            ax.plot(t.n, t[col], 'o-', color=C['data'])
            if col == 'bias':
                ax.axhline(0, color=C['ref'], lw=1, ls='--')
            ax.set_xscale('log'); ax.set_xticks(list(t.n)); ax.set_xticklabels([str(v) for v in t.n]); ax.minorticks_off()
            ax.set(xlabel='объём подвыборки $m$ (лог. шкала)', ylabel=f'{what}, {unit}')
            L.letter(ax, k, f'{key}: {"разброс" if col == "sd" else "смещение"}'); k += 1
    return fig


def mwu(a, b, alt):
    """Манн–Уитни: статистика, p-value и δ Клиффа."""
    m = stats.mannwhitneyu(a, b, alternative=alt)
    return m.statistic, m.pvalue, 2 * m.statistic / (len(a) * len(b)) - 1


def hypothesis_row(h, h0, h1, test, stat, p, eff, alpha=0.05):
    return {'гипотеза': h, 'H0': h0, 'H1': h1, 'критерий': test, 'статистика': stat, 'p': p, 'эффект': eff,
            'решение': 'H0 отвергается' if p < alpha else 'H0 не отвергается'}


def hypotheses_piezo(piezo, D1, pb1, p1, rep13):
    """Шесть гипотез о пьезокоэффициенте (H1.1–H1.6)."""
    x1 = piezo.eij_max.values
    H = []
    add = lambda *a: H.append(hypothesis_row(*a))
    lgx = np.log10(x1)
    k2 = stats.normaltest(lgx)
    add('H1.1', 'lg e_ij_max ~ N(μ,σ²)', 'не нормален', 'D’Agostino K²', k2.statistic, k2.pvalue, f'skew={stats.skew(lgx):.2f}')
    add('H1.2', 'e_ij_max ~ Lomax(α, λ)', 'иное распределение', 'КС, парам. бутстреп', D1, pb1, f'α̂={p1[0]:.2f}')
    st, p, cl = mwu(piezo.eij_max[piezo.toxic], piezo.eij_max[~piezo.toxic], 'greater')
    add('H1.3', 'e_ij_max (с токсичными эл-тами) = (без них)', 'у токсичных больше', 'Манн–Уитни (одностор.)', st, p, f'δ Клиффа={cl:.2f}; медианы {piezo.eij_max[piezo.toxic].median():.2f}/{piezo.eij_max[~piezo.toxic].median():.2f}')
    st, p, cl = mwu(piezo.eij_max[piezo.polar], piezo.eij_max[~piezo.polar], 'two-sided')
    add('H1.4', 'e_ij_max (полярные группы) = (неполярные)', '≠', 'Манн–Уитни', st, p, f'δ Клиффа={cl:.2f}')
    st, p, cl = mwu(piezo.eij_max[piezo.has_O], piezo.eij_max[~piezo.has_O], 'two-sided')
    add('H1.5', 'e_ij_max (оксиды) = (не оксиды)', '≠', 'Манн–Уитни', st, p, f'δ Клиффа={cl:.2f}')
    add('H1.6', 'ρ Спирмена(e_ij_max, V/N) = 0', 'ρ ≠ 0', 'Спирмен', rep13['Spearman ρ'], rep13['p (Spearman)'], f'ρ={rep13["Spearman ρ"]:.2f}')
    return pd.DataFrame(H)


def hypotheses_diel(diel, D2, pb2, p2):
    """Пять гипотез о показателе преломления (H2.1–H2.5)."""
    x2 = diel.n.values
    err = (diel.n - diel.n_pred_modnet).abs()
    H = []
    add = lambda *a: H.append(hypothesis_row(*a))
    k2 = stats.normaltest(np.log10(x2))
    add('H2.1', 'lg n ~ N(μ,σ²)', 'не нормален', 'D’Agostino K²', k2.statistic, k2.pvalue, f'skew={stats.skew(np.log10(x2)):.2f}')
    add('H2.2', 'n ~ GEV', 'иное распределение', 'КС, парам. бутстреп', D2, pb2, f'ξ̂={-p2[0]:.2f}')
    kw = stats.kruskal(*[g.n.values for _, g in diel.groupby('fold')])
    add('H2.3', 'распределения n в 5 фолдах одинаковы', 'различаются', 'Краскел–Уоллис', kw.statistic, kw.pvalue, f'размах медиан фолдов={diel.groupby("fold").n.median().max() - diel.groupby("fold").n.median().min():.4f}')
    sp = stats.spearmanr(diel.n, err)
    add('H2.4', 'ρ Спирмена(n, |ошибка модели|) = 0', 'ρ ≠ 0', 'Спирмен', sp[0], sp[1], f'ρ={sp[0]:.2f}')
    diff = diel.n - diel.n_pred_modnet
    nonzero = diff[diff != 0]
    tt = stats.binomtest(int((nonzero > 0).sum()), len(nonzero), p=0.5)
    add('H2.5', 'P(n − n_pred > 0 | ошибка ≠ 0) = 0.5', '≠ 0.5', 'Знаковый биномиальный',
        tt.statistic, tt.pvalue, f'медиана ошибки={np.median(diff):.3f}; нулей={(diff == 0).sum()}')
    return pd.DataFrame(H)


def summary_row(name, unit, x, dist, par, pval, ft):
    return {'величина': name, 'ед.': unit, 'N': len(x), 'mean': x.mean(), 'SD': x.std(ddof=1), 'median': np.median(x), 'skew': stats.skew(x),
            'ex.kurt': stats.kurtosis(x), 'лучшее по AIC': ft.dist.iloc[0], 'проверенное семейство': dist, 'параметры (MLE)': tuple(np.round(par, 3)), 'p (бутстреп КС)': pval}


def modnet_quality(diel):
    """Качество сохранённых предсказаний MODNet по фолдам и по всем данным."""
    r2 = lambda d: 1 - ((d.n - d.n_pred_modnet) ** 2).sum() / ((d.n - d.n.mean()) ** 2).sum()
    met = lambda d: pd.Series({'N': len(d), 'R² (MODNet)': r2(d), 'MAE': (d.n - d.n_pred_modnet).abs().mean(),
                               'Спирмен(n, предсказание)': stats.spearmanr(d.n, d.n_pred_modnet)[0]})
    folds = diel.groupby('fold')[['n', 'n_pred_modnet']].apply(met)
    folds.index = [f'фолд {i}' for i in folds.index]
    folds.loc['все данные'] = met(diel)
    return folds


def log_zscore_row(x):
    """Строка «lg x → z-оценка» для таблицы нормализации."""
    lg = np.log10(x); z = (lg - lg.mean()) / lg.std(ddof=1)
    return [z.mean(), z.std(ddof=1), z.min(), z.max(), stats.skew(z), stats.kurtosis(z), stats.normaltest(z).pvalue]


def polymorph_table(df, target='eij_max'):
    """Разброс свойства среди структур с одинаковой формулой (формулы, встречающиеся не меньше двух раз)."""
    g = df.groupby('formula').filter(lambda d: len(d) >= 2).groupby('formula')[target].agg(['count', 'min', 'max'])
    g['max/min'] = g['max'] / g['min']
    return g.sort_values('max/min', ascending=False)


def population_row(x):
    return pd.Series({'N': len(x), 'μ': x.mean(), 'σ (N)': x.std(ddof=0), 'S (N−1)': x.std(ddof=1), 'медиана': np.median(x),
                      'skew': stats.skew(x), 'ex.kurt': stats.kurtosis(x)}, name='генеральная').to_frame().T


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
DIST_RU = {'burr12': 'Burr XII', 'lomax': 'Ломакс', 'genpareto': 'обобщённое Парето (GPD)', 'fisk': 'лог-логистическое (Fisk)',
           'weibull_min': 'Вейбулла', 'lognorm': 'логнормальное', 'gamma': 'гамма', 'invgauss': 'обратное гауссово',
           'expon': 'экспоненциальное', 'genextreme': 'GEV', 'johnsonsu': 'Johnson SU', 'skewnorm': 'скошенное нормальное',
           'norm': 'нормальное', 'invgamma': 'обратное гамма'}


def summary_table(x, unit):
    """Описательные статистики с русскими названиями и единицами измерения (расчёт — в summary)."""
    s = summary(x)
    u2 = unit or '—'
    rows = [('N, число записей', s['n'], 'шт.'), ('среднее', s['mean'], u2), ('SD (ddof=1)', s['std (ddof=1)'], u2),
            ('минимум', s['min'], u2), ('Q1, первый квартиль', s['Q1'], u2), ('медиана', s['median'], u2),
            ('Q3, третий квартиль', s['Q3'], u2), ('максимум', s['max'], u2), ('IQR = Q3 − Q1', s['IQR'], u2),
            ('асимметрия', s['skew'], '—'), ('эксцесс (избыточный)', s['ex.kurtosis'], '—'),
            ('CV = SD / среднее', s['CV'], '—'), ('выбросов по правилу 1.5·IQR', s['выбросы (1.5·IQR)'], 'шт.')]
    return pd.DataFrame(rows, columns=['характеристика', 'значение', 'ед.']).set_index('характеристика')


def label_terciles(ct, row_name, col_name):
    """Подписи для таблицы сопряжённости 3×3 по терцилям."""
    names = ['нижняя треть', 'средняя треть', 'верхняя треть']
    out = ct.copy()
    out.index = pd.Index(names[:len(out.index)], name=row_name)
    out.columns = pd.Index(names[:len(out.columns)], name=col_name)
    return out
