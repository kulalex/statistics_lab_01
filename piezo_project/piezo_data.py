import json, re
from pathlib import Path
import numpy as np, pandas as pd
from mendeleev import element

DATA = Path(__file__).parent / 'data'
POLAR = {'1', '2', 'm', 'mm2', '4', '4mm', '3', '3m', '6', '6mm'}     # полярные точечные группы
TOXIC = {'Pb', 'Cd', 'Hg', 'Tl', 'As', 'Be'}                          # рабочее определение «токсичных» элементов
_cache = {}


def elems(formula):
    """Разбор формулы с повторяющимися элементами и вложенными скобками."""
    tokens = re.findall(r'[A-Z][a-z]?|\d+(?:\.\d+)?|[()]', formula)
    if ''.join(tokens) != formula:
        raise ValueError(f'Неподдерживаемая формула: {formula}')
    stack = [{}]
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t == '(':
            stack.append({}); i += 1; continue
        if t == ')':
            if len(stack) == 1:
                raise ValueError(f'Лишняя скобка: {formula}')
            group = stack.pop(); i += 1
        elif re.fullmatch(r'[A-Z][a-z]?', t):
            group = {t: 1.0}; i += 1
        else:
            raise ValueError(f'Неожиданный токен: {formula}')
        factor = 1.0
        if i < len(tokens) and re.fullmatch(r'\d+(?:\.\d+)?', tokens[i]):
            factor = float(tokens[i]); i += 1
        for sym, count in group.items():
            stack[-1][sym] = stack[-1].get(sym, 0.0) + factor * count
    if len(stack) != 1 or not stack[0]:
        raise ValueError(f'Незакрытая/пустая формула: {formula}')
    return stack[0]


def _el(sym):
    if sym not in _cache:
        e = element(sym); _cache[sym] = (e.atomic_number, e.atomic_weight, e.en_pauling if e.en_pauling is not None else np.nan)
    return _cache[sym]


def crystal_system(sg):
    for lim, name in ((2, 'триклинная'), (15, 'моноклинная'), (74, 'ромбическая'), (142, 'тетрагональная'),
                      (167, 'тригональная'), (194, 'гексагональная'), (230, 'кубическая')):
        if sg <= lim: return name


def load_piezo():
    with (DATA / 'piezo.json').open() as handle:
        raw = json.load(handle)
    rows = []
    for x in raw:
        m = x['meta']; el = elems(m['formula']); tot = sum(el.values())
        z, mass, en = zip(*[(_el(s)[0], _el(s)[1], _el(s)[2]) for s in el]); w = np.array(list(el.values())) / tot
        en = np.array(en)
        rows.append({'id': m['material_id'], 'formula': m['formula'], 'sg': m['space_group'], 'pg': m['point_group'],
                     'nsites': m['nsites'], 'volume': m['volume'], 'eij_max': x['eij_max'],
                     'n_elem': len(el), 'Z_mean': (w * z).sum(), 'mass_mean': (w * mass).sum(),
                     'EN_mean': np.sum(w * en), 'EN_range': np.nanmax(en) - np.nanmin(en),
                     'has_O': 'O' in el, 'has_halogen': bool(set(el) & {'F', 'Cl', 'Br', 'I'}),
                     'toxic': bool(set(el) & TOXIC), 'polar': m['point_group'] in POLAR,
                     'system': crystal_system(m['space_group'])})
    df = pd.DataFrame(rows)
    df['vol_atom'] = df.volume / df.nsites
    df['system_code'] = df.system.astype('category').cat.codes
    return df


def load_dielectric():
    return pd.read_csv(DATA / 'matbench_dielectric_n_folds.csv')


def validate_data(piezo, diel):
    """Проверки прекращают расчёт при повреждённых данных; строки не удаляются."""
    records = []
    for name, df, key, columns, expected in [
        ('piezo', piezo, 'id', ['eij_max', 'volume', 'nsites', 'vol_atom'], 941),
        ('dielectric', diel, 'id', ['n', 'n_pred_modnet'], 4764),
    ]:
        if len(df) != expected:
            raise ValueError(f'{name}: ожидалось {expected} строк, получено {len(df)}')
        if key not in df or df[key].isna().any() or df[key].duplicated().any():
            raise ValueError(f'{name}: отсутствующие/повторные id')
        vals = df[columns].to_numpy(dtype=float)
        if not np.isfinite(vals).all() or (vals <= 0).any():
            raise ValueError(f'{name}: значения должны быть конечными и положительными')
        records.append({'датасет': name, 'N': len(df), 'повторных id': int(df[key].duplicated().sum()),
                        'пропусков в целевых полях': int(df[columns].isna().sum().sum()),
                        'строк с пропуском EN': int(df.EN_mean.isna().sum()) if 'EN_mean' in df else 0})
    if set(diel.fold.unique()) != set(range(5)):
        raise ValueError('Ожидались пять фолдов 0–4')
    return pd.DataFrame(records)
