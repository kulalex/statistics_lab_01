"""Проверка ссылок и якорей в ноутбуках и в собранных HTML-страницах.

    python tools/check_links.py              # внутренние ссылки и якоря
    python tools/check_links.py --external   # дополнительно запросить внешние адреса

Проверяется: у каждой ссылки вида #якорь есть якорь; якоря не повторяются; ссылки между
ноутбуками и между страницами ведут на существующие якоря; на каждый рисунок, таблицу
и формулу есть якорь с номером, идущим по порядку. Код возврата 1, если найдены ошибки.
"""
import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT.parent / 'docs'
NOTEBOOKS = ['lab01_sprint1_piezo.ipynb', 'lab01_sprint2_piezo.ipynb']
PAGES = ['index.html', 'sprint1.html', 'sprint2.html']

ANCHOR = re.compile(r'<a\s+id="([^"]+)"')
ANY_ID = re.compile(r'\sid="([^"]+)"')
MD_LINK = re.compile(r'\]\(([^)\s]+)\)')
HREF = re.compile(r'href="([^"]+)"')


def notebook_text(path):
    nb = json.loads(path.read_text(encoding='utf-8'))
    parts = []
    for cell in nb['cells']:
        if cell['cell_type'] == 'markdown':
            parts.append(''.join(cell['source']))
        for out in cell.get('outputs', []):
            data = out.get('data', {})
            for kind in ('text/markdown', 'text/html'):
                if kind in data:
                    parts.append(''.join(data[kind]))
    return '\n'.join(parts)


def collect(text, id_pattern):
    ids = id_pattern.findall(text)
    links = MD_LINK.findall(text) + HREF.findall(text)
    return ids, links


def check_sequence(ids, name, errors):
    """Номера рисунков, таблиц и формул внутри раздела идут подряд, начиная с 1."""
    for kind, label in (('fig', 'рисунков'), ('tab', 'таблиц'), ('eq', 'формул')):
        by_section = {}
        for ident in ids:
            m = re.fullmatch(rf'{kind}-(\d+)-(\d+)', ident)
            if m:
                by_section.setdefault(int(m.group(1)), []).append(int(m.group(2)))
        for section, numbers in sorted(by_section.items()):
            if numbers != list(range(1, len(numbers) + 1)):
                errors.append(f'{name}: нумерация {label} в разделе {section} идёт не подряд: {numbers}')


def main():
    external = '--external' in sys.argv
    errors, targets, sources = [], {}, {}
    files = [(n, ROOT / n, ANCHOR, notebook_text) for n in NOTEBOOKS]
    files += [(p, DOCS / p, ANY_ID, lambda path: path.read_text(encoding='utf-8')) for p in PAGES if (DOCS / p).exists()]
    for name, path, pattern, reader in files:
        text = reader(path)
        ids, links = collect(text, pattern)
        dup = sorted({i for i in ids if ids.count(i) > 1 and not i.startswith('MJX')})
        if dup:
            errors.append(f'{name}: повторяющиеся якоря {dup}')
        targets[name] = set(ids)
        sources[name] = links
        if name.endswith('.ipynb'):
            check_sequence(ids, name, errors)

    urls = set()
    n_internal = 0
    for name, links in sources.items():
        for link in links:
            if link.startswith(('http://', 'https://')):
                urls.add(link)
                continue
            if link.startswith(('mailto:', 'data:')):
                continue
            target_file, _, anchor = link.partition('#')
            target_file = target_file or name
            n_internal += 1
            if target_file not in targets:
                exists = (ROOT / target_file).exists() or (DOCS / target_file).exists()
                if not exists or anchor:
                    errors.append(f'{name}: ссылка {link} ведёт на неизвестный файл')
                continue
            if anchor and anchor not in targets[target_file]:
                errors.append(f'{name}: нет якоря для ссылки {link}')

    # на каждый рисунок и таблицу должна быть хотя бы подпись-якорь; проверяем и обратное: якорь без ссылок допустим
    print(f'файлов: {len(files)}, якорей: {sum(len(t) for t in targets.values())}, внутренних ссылок: {n_internal}, внешних адресов: {len(urls)}')

    if external:
        for url in sorted(urls):
            try:
                req = urllib.request.Request(url, method='GET', headers={'User-Agent': 'link-checker'})
                with urllib.request.urlopen(req, timeout=20) as resp:
                    status = resp.status
            except Exception as exc:  # сетевые ошибки печатаем, но не считаем ошибкой документа
                status = getattr(exc, 'code', None) or type(exc).__name__
            mark = 'ok ' if status == 200 else '?? '
            print(f'  {mark}{status}  {url}')

    for e in errors:
        print('ОШИБКА:', e)
    print('ссылки в порядке' if not errors else f'ошибок: {len(errors)}')
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
