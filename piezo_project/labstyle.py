"""Единое оформление графиков и таблиц для обоих спринтов.

Цвет закреплён за ролью, а не за порядком линий на рисунке:
данные всегда синие, теоретическая модель оранжевая, генератор зелёный.
Роли продублированы типом линии, чтобы рисунок читался и в чёрно-белой печати.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from IPython.display import display

C = {
    'data': '#2a78d6',    # наблюдения из датасета
    'model': '#eb6834',   # подогнанное распределение (теория)
    'gen': '#1baf7a',     # выборка из генератора
    'gen2': '#e87ba4',    # второй генератор на том же рисунке
    'ref': '#52514e',     # опорные линии: ноль, диагональ, уровень
    'ink': '#0b0b0b',
    'muted': '#8a8985',
    'grid': '#e4e3df',
}
# старые имена ролей из statlib
C['fit'] = C['model']
C['alt'] = C['gen']
C['grey'] = C['ref']

LS = {'data': '-', 'model': '-', 'gen': '--', 'gen2': ':'}

FULL = 10.0    # ширина рисунка на всю колонку, дюймы
HALF = 5.6     # ширина одиночного графика
PANEL = 3.3    # высота одной строки панелей

LETTERS = 'абвгдежзик'


def apply():
    """Включить общий стиль. Вызывается один раз в начале ноутбука."""
    plt.rcParams.update({
        'figure.dpi': 144, 'savefig.dpi': 144, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'figure.constrained_layout.use': True,
        'font.family': 'DejaVu Sans', 'font.size': 10, 'mathtext.fontset': 'dejavusans',
        'axes.titlesize': 10.5, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left', 'axes.titlepad': 8,
        'axes.labelsize': 10, 'axes.labelcolor': C['ink'], 'axes.edgecolor': C['muted'], 'axes.linewidth': 0.8,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.color': C['grid'], 'grid.linewidth': 0.8, 'axes.axisbelow': True,
        'xtick.color': C['ref'], 'ytick.color': C['ref'], 'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'lines.linewidth': 2.0, 'lines.markersize': 6,
        'legend.frameon': False, 'legend.fontsize': 9,
        'axes.formatter.use_mathtext': True,
    })
    pd.set_option('display.max_colwidth', None, 'display.max_columns', None, 'display.width', 200)


def plain_log(ax, which='x'):
    """На узкой логарифмической оси (меньше 2.5 декад) ставит обычные числа 1, 2, 5, 10, 20, 50 вместо степеней десяти."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
    axis = ax.xaxis if which == 'x' else ax.yaxis
    lo, hi = ax.get_xlim() if which == 'x' else ax.get_ylim()
    if lo <= 0 or np.log10(hi / lo) > 2.5:
        return
    ticks = [m * 10.0 ** k for k in range(int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi))) + 1) for m in (1, 2, 5)]
    ticks = [t for t in ticks if lo <= t <= hi]
    if len(ticks) < 2:
        return
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    axis.set_minor_formatter(NullFormatter())


def letter(ax, i, title):
    """Заголовок панели с буквой: «(а) Гистограмма»."""
    ax.set_title(f'({LETTERS[i]}) {title}')


_SUP = str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹')


def fmt(v, sig=4):
    """Число для таблицы: целые без дробной части, очень большие и малые как a·10ⁿ, остальные с sig значащими цифрами."""
    if isinstance(v, (bool, np.bool_)):
        return 'да' if v else 'нет'
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        v = float(v)
        if np.isnan(v):
            return '—'
        if np.isinf(v):
            return '∞' if v > 0 else '−∞'
        if v == 0:
            return '0'
        a = abs(v)
        if v.is_integer() and a < 1e6:
            return str(int(v)).replace('-', '−')
        if a >= 1e5 or a < 1e-3:
            mant, exp = f'{v:.{sig - 1}e}'.split('e')
            return f'{mant}·10{str(int(exp)).translate(_SUP)}'.replace('-', '−')
        return f'{v:.{sig}g}'.replace('-', '−')
    if isinstance(v, (tuple, list, np.ndarray)):
        return '(' + ', '.join(fmt(float(x) if isinstance(x, np.generic) else x, sig) for x in v) + ')'
    return str(v)


# единые русские названия столбцов для обоих спринтов (только для показа, в расчётах имена прежние)
RU = {'mean': 'среднее', 'median': 'медиана', 'std': 'SD', 'SD (sample)': 'SD', 'skew': 'асимметрия',
      'ex.kurt': 'эксцесс', 'kurt': 'эксцесс', 'min': 'минимум', 'max': 'максимум',
      'truth': 'цель θ', 'eps': 'ε = 0.1|θ|', 'mean_est': 'средняя оценка', 'bias': 'смещение', 'sd': 'SD оценки',
      'estimator': 'оценка', 'dist': 'семейство', 'params': 'параметры (MLE)', 'replace': 'с возвращением',
      'SE(median)': 'SE(медианы)', 'KS D': 'КС D', 'KS statistic': 'КС D', 'p-value': 'p', 'bootstrap p': 'p (бутстреп)',
      'KS D (описательно)': 'КС D до полного набора', 'RV': 'величина', 'k': 'число параметров k'}


def table(df, sig=4, rename=None, index=True):
    """Привести таблицу к виду для чтения: округлить числа, перевести названия столбцов."""
    if isinstance(df, pd.Series):
        df = df.to_frame()
    out = df.copy()
    out = out.apply(lambda col: col.map(lambda v: fmt(v, sig)))
    names = {**RU, **(rename or {})}
    out = out.rename(columns=names, index=names)
    out.index.name = names.get(out.index.name, out.index.name)
    out.columns.name = names.get(out.columns.name, out.columns.name)
    if index:                                   # значимый индекс становится первым столбцом
        first = out.index.name or ''
        out = out.reset_index()
        out.columns = [first] + list(out.columns[1:])
    else:
        out = out.reset_index(drop=True)
    out.columns.name = None
    return out


def show(df, sig=4, rename=None, index=True):
    """Показать таблицу без служебного столбца с номерами строк."""
    t = table(df, sig, rename, index)
    classes = ['rowhead'] if index else None
    display({'text/html': t.to_html(index=False, border=0, classes=classes),
             'text/plain': t.to_string(index=False)}, raw=True)


def flow(steps, ncols=4, width=FULL, box_h=1.0, title=None):
    """Схема хода исследования: блоки с номером раздела, стрелки слева направо и построчно.

    steps: список (номер, заголовок, пояснение, роль), роль: 'data' | 'model' | 'gen' | 'ref'.
    """
    nrows = int(np.ceil(len(steps) / ncols))
    gap_x, gap_y, box_w = 0.35, 0.55, 2.2
    W = ncols * box_w + (ncols - 1) * gap_x
    H = nrows * box_h + (nrows - 1) * gap_y
    fig, ax = plt.subplots(figsize=(width, width * (H + 0.3) / W))
    ax.set(xlim=(-0.05, W + 0.05), ylim=(-0.15, H + 0.15)); ax.axis('off')
    pos = []
    for k, (num, head, text, role) in enumerate(steps):
        r, c = divmod(k, ncols)
        x = c * (box_w + gap_x); y = H - (r + 1) * box_h - r * gap_y
        pos.append((x, y))
        ax.add_patch(FancyBboxPatch((x, y), box_w, box_h, boxstyle='round,pad=0,rounding_size=0.08',
                                    fc='white', ec=C[role], lw=1.6))
        ax.add_patch(FancyBboxPatch((x, y + box_h - 0.07), box_w, 0.07, boxstyle='square,pad=0', fc=C[role], ec='none'))
        ax.text(x + 0.12, y + box_h - 0.2, num, fontsize=8.5, color=C['ref'], va='top')
        ax.text(x + 0.12, y + box_h - 0.42, head, fontsize=9.5, weight='bold', color=C['ink'], va='top')
        ax.text(x + 0.12, y + 0.1, text, fontsize=8, color=C['ref'], va='bottom', linespacing=1.25)
    arrow = dict(arrowstyle='-|>', color=C['ref'], lw=1.2, shrinkA=2, shrinkB=2)
    for k in range(len(steps) - 1):
        (x0, y0), (x1, y1) = pos[k], pos[k + 1]
        if y0 == y1:
            ax.annotate('', (x1, y1 + box_h / 2), (x0 + box_w, y0 + box_h / 2), arrowprops=arrow)
        else:
            ym = y0 - gap_y / 2
            ax.plot([x0 + box_w / 2, x0 + box_w / 2, x1 + box_w / 2], [y0, ym, ym], color=C['ref'], lw=1.2)
            ax.annotate('', (x1 + box_w / 2, y1 + box_h), (x1 + box_w / 2, ym), arrowprops=arrow)
    if title:
        ax.set_title(title)
    return fig


def source(fn):
    """Показать исходный код функции в сворачиваемом блоке (в разделах «code» спринта 2)."""
    import html
    import inspect
    from IPython.display import HTML
    code = html.escape(inspect.getsource(fn))
    display(HTML(f'<details class="src"><summary>Исходный код функции <code>{fn.__name__}</code> '
                 f'(модуль <code>{fn.__module__}.py</code>)</summary>\n<pre><code class="language-python">{code}</code></pre>\n</details>'))
