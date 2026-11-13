"""Спринт 2: генераторы случайных величин, их проверки и графики.

Генераторы перенесены из ноутбука без изменения вычислений.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import labstyle as L

SEED = 2026
DATA_DIR = Path(__file__).parent / 'data'
N_MC = 499


# ---------------------------------------------------------------- данные

def load_samples(data_dir=DATA_DIR):
    """Три выборки: x1 = e_ij_max, x2 = n, x3 = V/N. Берутся только конечные положительные значения."""
    piezo_path = Path(data_dir) / 'piezo.json'
    diel_path = Path(data_dir) / 'matbench_dielectric_n_folds.csv'
    if not piezo_path.exists() or not diel_path.exists():
        raise FileNotFoundError('Положите папку data рядом с ноутбуком. '
                                'В ней нужны piezo.json и matbench_dielectric_n_folds.csv')
    with piezo_path.open(encoding='utf-8') as f:
        raw_piezo = json.load(f)
    piezo_rows = []
    for row in raw_piezo:
        meta = row['meta']
        if row.get('eij_max') is not None and meta.get('volume') and meta.get('nsites'):
            piezo_rows.append({'eij_max': float(row['eij_max']),
                               'vol_atom': float(meta['volume']) / float(meta['nsites'])})
    piezo = pd.DataFrame(piezo_rows)
    diel = pd.read_csv(diel_path)
    x1 = piezo.loc[np.isfinite(piezo.eij_max) & (piezo.eij_max > 0), 'eij_max'].to_numpy()
    x2 = diel.loc[np.isfinite(diel.n) & (diel.n > 0), 'n'].to_numpy()
    x3 = piezo.loc[np.isfinite(piezo.vol_atom) & (piezo.vol_atom > 0), 'vol_atom'].to_numpy()
    return x1, x2, x3


# ---------------------------------------------------------------- генераторы

def lomax_inverse(n, alpha, scale, rng=None):
    """Lomax через обратную функцию распределения."""
    if not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError("n должно быть целым неотрицательным")
    if not np.isfinite(alpha) or alpha <= 0 or not np.isfinite(scale) or scale <= 0:
        raise ValueError("alpha и scale должны быть положительными конечными числами")
    rng = np.random.default_rng() if rng is None else rng
    u = rng.random(n)
    return scale * np.expm1(-np.log1p(-u) / alpha)


def lomax_gamma_mixture(n, alpha, scale, rng=None):
    """Lomax как смесь экспоненциальных: T ~ Gamma(alpha, rate=1), X | T ~ Exp(rate=T/scale)."""
    if not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError("n должно быть целым неотрицательным")
    if not np.isfinite(alpha) or alpha <= 0 or not np.isfinite(scale) or scale <= 0:
        raise ValueError("alpha и scale должны быть положительными конечными числами")
    rng = np.random.default_rng() if rng is None else rng
    latent_rate = rng.gamma(shape=alpha, scale=1.0, size=n)
    latent_rate = np.maximum(latent_rate, np.finfo(float).tiny)
    return rng.exponential(scale=scale / latent_rate)


def gev_inverse(n, c, loc, scale, rng=None):
    """GEV через квантиль SciPy; c = -xi."""
    if n < 0 or int(n) != n or not np.isfinite(scale) or scale <= 0:
        raise ValueError("Некорректные n или scale")
    rng = np.random.default_rng() if rng is None else rng
    u = np.clip(rng.random(int(n)), np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0))
    return stats.genextreme.ppf(u, c, loc=loc, scale=scale)


def standard_normal_box_muller(n, rng=None):
    """n независимых значений N(0,1) методом Бокса–Мюллера."""
    if n < 0 or int(n) != n:
        raise ValueError("n должно быть целым неотрицательным")
    rng = np.random.default_rng() if rng is None else rng
    pairs = (int(n) + 1) // 2
    u1 = np.maximum(rng.random(pairs), np.finfo(float).tiny)
    u2 = rng.random(pairs)
    radius = np.sqrt(-2.0 * np.log(u1))
    z = np.empty(2 * pairs)
    z[0::2] = radius * np.cos(2 * np.pi * u2)
    z[1::2] = radius * np.sin(2 * np.pi * u2)
    return z[:int(n)]


def shifted_lognormal_box_muller(n, shape, loc, scale, rng=None):
    """X = loc + exp(Y), Y ~ N(log(scale), shape^2)."""
    if shape <= 0 or scale <= 0 or not np.isfinite(shape + loc + scale):
        raise ValueError("shape и scale должны быть положительными; параметры конечными")
    rng = np.random.default_rng() if rng is None else rng
    z = standard_normal_box_muller(n, rng)
    return loc + np.exp(np.log(scale) + shape * z)


# ---------------------------------------------------------------- автотесты

def contract_tests(fn, args, lower=None, strict=False, bad_args=None, n=500, seed=777):
    """Проверки контракта генератора. Возвращает таблицу «проверка → результат», при нарушении падает."""
    out_a = fn(n, *args, rng=np.random.default_rng(seed))
    out_b = fn(n, *args, rng=np.random.default_rng(seed))
    rows = []
    assert out_a.shape == (n,)
    rows.append(('размер результата равен запрошенному', f'{out_a.shape[0]} из {n}'))
    assert np.all(np.isfinite(out_a))
    rows.append(('все значения конечны', 'да'))
    assert np.array_equal(out_a, out_b), f"Воспроизводимость не выполнена: {fn.__name__}"
    rows.append(('одинаковый seed даёт одинаковую последовательность', 'да'))
    if bad_args is not None:
        try:
            fn(*bad_args, rng=np.random.default_rng(1))
        except ValueError:
            rows.append((f'некорректные аргументы {bad_args} отклоняются', 'ValueError'))
        else:
            raise AssertionError("Ожидалась ValueError для некорректных параметров")
    return rows


def support_test(sample, lower, strict=False, tol=0.0):
    """Все значения лежат в носителе распределения."""
    ok = np.all(sample > lower) if strict else np.all(sample >= lower - tol)
    assert ok
    sign = '>' if strict else '≥'
    return (f'все {len(sample)} значений {sign} {L.fmt(float(lower))} (нижняя граница носителя)', 'да')


def normal_moments_test(z):
    """Базовые моменты стандартного нормального датчика."""
    assert abs(z.mean()) < 0.02
    assert abs(z.std(ddof=1) - 1) < 0.02
    return [('|среднее| < 0.02', L.fmt(float(z.mean()))),
            ('|SD − 1| < 0.02', L.fmt(float(z.std(ddof=1)))),
            ('асимметрия (для справки)', L.fmt(float(stats.skew(z))))]


def tests_table(rows):
    return pd.DataFrame(rows, columns=['проверка', 'результат'])


def pit_row(name, pit):
    res = stats.kstest(pit, 'uniform')
    return {'генератор': name, 'KS D': res.statistic, 'p-value': res.pvalue}


# ---------------------------------------------------------------- сравнение с данными

def describe_sample(name, arr):
    arr = np.asarray(arr)
    return {"Набор": name, "N": len(arr), "mean": np.mean(arr), "median": np.median(arr),
            "SD (sample)": np.std(arr, ddof=1), "q90": np.quantile(arr, .90),
            "q95": np.quantile(arr, .95), "q99": np.quantile(arr, .99)}


def theoretical_row(name, dist, params):
    q = dist.ppf([.90, .95, .99], *params)
    mean, var = dist.stats(*params, moments="mv")
    return {"Набор": name, "N": "теория", "mean": float(mean), "median": float(dist.ppf(.5, *params)),
            "SD (sample)": float(np.sqrt(var)) if np.isfinite(var) and var >= 0 else np.nan,
            "q90": q[0], "q95": q[1], "q99": q[2]}


def ks_rows(pairs):
    rows = []
    for label, a, b in pairs:
        res = stats.ks_2samp(a, b)
        rows.append({"Сравнение": label, "KS D": res.statistic, "p-value": res.pvalue})
    return pd.DataFrame(rows)


def gof_bootstrap(dist, data, known_params=None, statistic="ks", n_mc=N_MC, seed=SEED):
    known_params = {} if known_params is None else known_params
    return stats.goodness_of_fit(dist, np.asarray(data), known_params=known_params,
                                 statistic=statistic, n_mc_samples=n_mc,
                                 rng=np.random.default_rng(seed))


# ---------------------------------------------------------------- состоятельность и скорость

def consistency_experiment(data, alpha_hat, sizes=(20, 40, 80, 160, 320, 640, 850), reps=250, seed=SEED):
    rng = np.random.default_rng(seed)
    pop_mean = float(np.mean(data)); pop_median = float(np.median(data))
    pop_q90 = float(np.quantile(data, .90))
    rows = []
    for n in sizes:
        vals = {"mean": [], "median": [], "q90": [], "alpha_MLE": []}
        alpha_fail = 0
        for _ in range(reps):
            sample = rng.choice(data, size=n, replace=False)
            vals["mean"].append(np.mean(sample))
            vals["median"].append(np.median(sample))
            vals["q90"].append(np.quantile(sample, .90))
            try:
                aa = stats.lomax.fit(sample, floc=0)[0]
                if np.isfinite(aa) and aa < 1e6: vals["alpha_MLE"].append(aa)
                else: alpha_fail += 1
            except Exception:
                alpha_fail += 1
        truths = {"mean": pop_mean, "median": pop_median, "q90": pop_q90, "alpha_MLE": alpha_hat}
        for key, estimates in vals.items():
            e = np.asarray(estimates)
            rows.append({"n": n, "оценка": key, "цель": truths[key],
                         "средняя оценка": np.mean(e) if len(e) else np.nan,
                         "bias": np.mean(e) - truths[key] if len(e) else np.nan,
                         "SD оценок": np.std(e, ddof=1) if len(e) > 1 else np.nan,
                         "RMSE": np.sqrt(np.mean((e - truths[key]) ** 2)) if len(e) else np.nan,
                         "доля неудачных alpha MLE": alpha_fail / reps if key == "alpha_MLE" else np.nan})
    return pd.DataFrame(rows)


def benchmark(fn, args, n=500_000, repeats=5, seed=SEED):
    times = []
    for i in range(repeats):
        rng = np.random.default_rng(seed + i)
        t0 = time.perf_counter(); values = fn(n, *args, rng=rng); times.append(time.perf_counter() - t0)
        assert len(values) == n
    return {"алгоритм": fn.__name__, "n": n, "повторов": repeats,
            "медианное время, с": np.median(times), "значений/с": n / np.median(times)}


# ---------------------------------------------------------------- графики

def ecdf(a):
    s = np.sort(a)
    return s, np.arange(1, len(s) + 1) / len(s)


def _count(n):
    return f'{n:,}'.replace(',', ' ')


def pit_figure(pit, name, z=None, bins=35):
    """Гистограмма PIT-значений F(X) против равномерной плотности; для Бокса–Мюллера рядом гистограмма Z."""
    ncol = 1 if z is None else 2
    fig, axes = plt.subplots(1, ncol, figsize=(L.HALF if ncol == 1 else L.FULL, L.PANEL), squeeze=False)
    ax = axes[0, 0]
    ax.hist(pit, bins=bins, density=True, color=L.C['gen'], alpha=.75, edgecolor='white', label=f'генератор, N = {_count(len(pit))}')
    ax.axhline(1, ls='--', color=L.C['model'], label='равномерная плотность U(0, 1)')
    ax.set(xlabel='$F(X)$, безразмерная', ylabel='плотность вероятности', xlim=(0, 1), ylim=(0, 1.35))
    ax.legend(loc='upper center', ncol=1)
    if z is None:
        ax.set_title(f'PIT: {name}')
    else:
        L.letter(ax, 0, f'PIT: {name}')
        ax = axes[0, 1]
        ax.hist(z, bins=50, density=True, color=L.C['gen'], alpha=.75, edgecolor='white', label=f'Бокс–Мюллер, N = {_count(len(z))}')
        zz = np.linspace(-4, 4, 300)
        ax.plot(zz, stats.norm.pdf(zz), color=L.C['model'], label='плотность N(0, 1)')
        ax.set(xlabel='$Z$, безразмерная', ylabel='плотность вероятности', ylim=(0, 0.52))
        ax.legend(loc='upper left')
        L.letter(ax, 1, 'промежуточная нормальная величина')
    return fig


def ecdf_figure(panels):
    """Данные против выборок из генераторов: сверху функция распределения, снизу хвост P(X > x) в лог-лог шкале.

    panels: список (данные, [(выборка, подпись, роль), ...], заголовок, подпись оси x, логарифмическая ли ось x сверху).
    """
    fig, axes = plt.subplots(2, len(panels), figsize=(L.FULL, 2 * L.PANEL + 0.9))
    for i, (real, sims, title, xlabel, logx) in enumerate(panels):
        top, bot = axes[0, i], axes[1, i]
        series = [(real, f'данные, N = {len(real)}', 'data')] + list(sims)
        for arr, name, role in series:
            xx, yy = ecdf(arr)
            style = dict(color=L.C[role], ls=L.LS[role], lw=2.6 if role == 'data' else 1.6)
            top.step(xx, yy, where='post', label=name, **style)
            keep = yy < 1
            bot.step(xx[keep], 1 - yy[keep], where='post', **style)
        if logx:
            top.set_xscale('log')
            top.set_xlim(np.quantile(real, .002) / 2, np.max(real) * 3)
            L.plain_log(top)
        else:
            top.set_xlim(0, np.max(real) * 1.15)
        top.set_xlabel(xlabel + (' (лог. шкала)' if logx else ''))
        L.letter(top, i, title)
        bot.set_xscale('log'); bot.set_yscale('log')
        bot.set_xlim(np.quantile(real, .5), np.max(real) * 4)
        bot.set_ylim(0.3 / len(real), 1)
        L.plain_log(bot)
        bot.set_xlabel(xlabel + ' (лог. шкала)')
        L.letter(bot, i + len(panels), 'хвост: ' + title)
        bot.legend(*top.get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(0.5, -0.26), fontsize=8.5)
    axes[0, 0].set_ylabel('эмпирическая $F(x)$')
    axes[1, 0].set_ylabel('$P(X > x)$ (лог. шкала)')
    return fig


UNITS = {'mean': 'Кл/м²', 'median': 'Кл/м²', 'q90': 'Кл/м²', 'alpha_MLE': 'безразмерная'}
EST_NAMES = {'mean': 'среднее', 'median': 'медиана', 'q90': 'квантиль 0.9', 'alpha_MLE': r'$\hat\alpha$ (MLE)'}


def consistency_figure(tab):
    """RMSE и разброс оценок в зависимости от объёма подвыборки, по панели на оценку (у оценок разные единицы)."""
    keys = list(dict.fromkeys(tab['оценка']))
    fig, axes = plt.subplots(1, len(keys), figsize=(L.FULL, L.PANEL))
    for i, (ax, key) in enumerate(zip(axes, keys)):
        g = tab[tab['оценка'] == key]
        ax.plot(g.n, g.RMSE, 'o-', color=L.C['data'], label='RMSE')
        ax.plot(g.n, g['SD оценок'], 's--', color=L.C['ref'], ms=4, lw=1.2, label='SD между повторами')
        ax.set_xscale('log', base=2); ax.set_yscale('log')
        ax.set_xticks([20, 40, 80, 160, 320, 850]); ax.set_xticklabels(['20', '40', '80', '160', '320', '850']); ax.minorticks_off()
        L.plain_log(ax, 'y')
        ax.set_xlabel('$m$ (лог. шкала)')
        ax.set_ylabel(f'ошибка, {UNITS[key]} (лог. шкала)')
        L.letter(ax, i, EST_NAMES[key])
    axes[-1].legend(loc='upper right', fontsize=8)
    return fig


def param_display(par1, par2, par3):
    """Таблица оценённых параметров с обозначениями из текста и единицами измерения."""
    a, _, lam = par1
    c, mu, sigma = par2
    s, theta, scale = par3
    f = L.fmt
    rows = [
        ('X₁ = e_ij_max', 'Ломакс', f'α̂ = {f(a)} (форма, безразмерная); λ̂ = {f(lam)} Кл/м² (масштаб)', 'α, loc = 0, scale = λ'),
        ('X₂ = n', 'GEV', f'ξ̂ = {f(-c)} (форма; в SciPy c = −ξ̂ = {f(c)}); μ̂ = {f(mu)} (положение); σ̂ = {f(sigma)} (масштаб); все безразмерные', 'c, loc = μ, scale = σ'),
        ('X₃ = V/N', 'сдвинутое логнормальное', f's = {f(s)} (форма, безразмерная); θ̂ = {f(theta)} Å³/атом (сдвиг); exp(μ_Y) = {f(scale)} Å³/атом, то есть μ_Y = {f(np.log(scale))}', 's, loc = θ, scale = exp(μ_Y)'),
    ]
    return pd.DataFrame(rows, columns=['величина', 'модель', 'оценки параметров (MLE)', 'порядок параметров в SciPy'])


COMPARE_ROWS = [  # (исходная метка, новая метка, единица)
    ('X1 реальные', 'X₁: данные', 'Кл/м²'), ('X1 inverse', 'X₁: генератор, обратная функция', 'Кл/м²'),
    ('X1 gamma mixture', 'X₁: генератор, гамма-смесь', 'Кл/м²'), ('X1 Lomax theory', 'X₁: теория, Ломакс', 'Кл/м²'),
    ('X2 реальные', 'X₂: данные', '—'), ('X2 GEV', 'X₂: генератор, GEV', '—'), ('X2 GEV theory', 'X₂: теория, GEV', '—'),
    ('X3 реальные', 'X₃: данные', 'Å³/атом'), ('X3 shifted lognormal', 'X₃: генератор, логнормальное', 'Å³/атом'),
    ('X3 lognormal theory', 'X₃: теория, логнормальное', 'Å³/атом'),
]


def comparison_display(comparison):
    """Та же таблица сравнения, сгруппированная по величинам: данные, генераторы, теория."""
    out = comparison.loc[[r[0] for r in COMPARE_ROWS]].copy()
    out.insert(0, 'ед.', [r[2] for r in COMPARE_ROWS])
    out.index = pd.Index([r[1] for r in COMPARE_ROWS], name='набор')
    out['N'] = out['N'].map(lambda v: '—' if v == 'теория' else v)
    return out
