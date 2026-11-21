"""Сборка HTML-версии отчёта из выполненных ноутбуков.

    python tools/build_html.py

Читает сохранённые выводы ноутбуков (ничего не пересчитывает) и пишет в ../docs
три самодостаточные страницы: index.html, sprint1.html, sprint2.html.
Код в ячейках свёрнут и раскрывается по клику. Картинки и стили встроены в файл;
из интернета подгружается только MathJax для формул.
"""
import base64
import html
import re
import struct
from pathlib import Path

import nbformat
from bs4 import BeautifulSoup
from nbconvert.filters.markdown_mistune import markdown2html_mistune
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT.parent / 'docs'
PAGES = [
    ('lab01_sprint1_piezo.ipynb', 'sprint1.html', 'Спринт 1'),
    ('lab01_sprint2_piezo.ipynb', 'sprint2.html', 'Спринт 2'),
]
LINKS = {nb: page for nb, page, _ in PAGES}
REPO_URL = 'https://github.com/kulalex/statistics_lab_01'
FIG_DPI = 144          # dpi рисунков в ноутбуке; на странице показываем в масштабе 96/144
MATHJAX = 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-svg.min.js'

CSS = r'''
:root{--bg:#fcfcfb;--surface:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--muted:#8a8985;--line:#e4e3df;
--accent:#1f5fae;--accent-bg:#eef4fc;--code-bg:#f6f5f2;--w-text:46rem;--w-wide:60rem;color-scheme:light}
*{box-sizing:border-box}
html{scroll-padding-top:4.2rem}
body{margin:0;background:var(--bg);color:var(--ink);font:17px/1.62 system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
-webkit-text-size-adjust:100%}
a{color:var(--accent);text-decoration-thickness:1px;text-underline-offset:2px}
a:hover{text-decoration-thickness:2px}
.top{position:sticky;top:0;z-index:5;display:flex;gap:.4rem 1.2rem;align-items:center;flex-wrap:wrap;padding:.6rem 1.2rem;
background:rgba(252,252,251,.94);backdrop-filter:blur(6px);border-bottom:1px solid var(--line);font-size:.92rem}
.top .brand{font-weight:700;color:var(--ink);text-decoration:none;margin-right:auto}
.top a.nav{color:var(--ink2);text-decoration:none;padding:.15rem .1rem;border-bottom:2px solid transparent}
.top a.nav:hover{color:var(--ink)}
.top a.nav[aria-current=page]{color:var(--ink);border-bottom-color:var(--accent);font-weight:600}
.top button{font:inherit;color:var(--ink2);background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:.2rem .7rem;cursor:pointer}
.top button:hover{color:var(--ink);border-color:var(--muted)}
.layout{display:grid;grid-template-columns:17.5rem minmax(0,1fr);gap:2.5rem;max-width:84rem;margin:0 auto;padding:0 1.2rem}
nav.toc{position:sticky;top:3.4rem;align-self:start;max-height:calc(100vh - 3.6rem);overflow:auto;padding:1.4rem 0 2rem;font-size:.86rem;line-height:1.35}
nav.toc .toc-title{font-size:.72rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:0 0 .6rem}
nav.toc ol{list-style:none;margin:0;padding:0}
nav.toc li{margin:0}
nav.toc a{display:block;padding:.22rem .5rem;border-left:2px solid var(--line);color:var(--ink2);text-decoration:none}
nav.toc a:hover{color:var(--ink);border-left-color:var(--accent);background:var(--accent-bg)}
nav.toc li.l2>a{font-weight:600;color:var(--ink);margin-top:.5rem}
nav.toc li.l3>a{padding-left:1.2rem}
details.toc-mobile{display:none}
main{min-width:0;padding:1.6rem 0 5rem}
main>*{max-width:var(--w-text)}
main>.wide,main>.out,main>.table-wrap,main>details.code{max-width:var(--w-wide)}
h1,h2,h3,h4{line-height:1.25;margin:2.2em 0 .6em;text-wrap:balance}
h1{font-size:2rem;margin-top:.4em;letter-spacing:-.01em}
h2{font-size:1.5rem;padding-top:1.4rem;border-top:1px solid var(--line)}
h3{font-size:1.18rem}
h4{font-size:1rem}
.num{color:var(--muted);font-weight:600;margin-right:.35em}
/* Print-editorial identity: ink, mineral blue, copper, warm paper. */
:root{--bg:#f8f7f3;--surface:#fff;--ink:#162e37;--ink2:#4e5e61;--muted:#718085;--line:#dce1df;
--accent:#126e79;--accent-bg:#e7f1f0;--code-bg:#f2f5f3;--w-text:49rem;--w-wide:69rem}
body{font-family:"Segoe UI",Inter,Arial,sans-serif;line-height:1.73}
html{scroll-behavior:smooth;scroll-padding-top:5.2rem}
::selection{background:#c1e2df;color:#102d37}
a{color:#086a78}
a:focus-visible,button:focus-visible,input:focus-visible{outline:3px solid #db8468;outline-offset:3px}
.top{background:#fdfcf9;min-height:4.25rem;border-bottom:1px solid #d1dad8;padding:.65rem clamp(1rem,3vw,2.6rem);gap:1rem 1.5rem}
.top .brand{display:flex;align-items:center;gap:.65rem;color:#173a43;font-size:.98rem;letter-spacing:-.02em}
.brand-symbol{display:grid;place-items:center;background:#173a43;color:#f7ded0;width:2.15rem;height:2.15rem;border-radius:4px;font-size:.79rem;letter-spacing:.02em}
.top a.nav{font-weight:620;color:#526568;padding:.42rem .18rem}
.top a.nav[aria-current=page]{color:#a54b31;border-bottom-color:#c9694c}
.top button{border-color:#b3c9c9;color:#23515a;border-radius:5px;padding:.36rem .74rem;background:#eff7f6;font-size:.8rem;font-weight:600}
.layout{grid-template-columns:18.7rem minmax(0,1fr);max-width:100rem;gap:clamp(2rem,4vw,5rem);padding:0 clamp(1rem,3vw,2.6rem)}
nav.toc{top:5rem;max-height:calc(100vh - 5rem);background:#173942;color:#ddecea;padding:1.7rem 1.05rem 2rem;border-radius:0 0 9px 9px;scrollbar-width:thin;scrollbar-color:#829a9b transparent;box-shadow:inset 0 4px #cb755a}
nav.toc .toc-title{color:#f1b39a;font-weight:700;font-size:.71rem;letter-spacing:.16em;margin:0 0 .9rem}
nav.toc a{color:#c1d6d6;border-left:2px solid #46656b;border-radius:0 4px 4px 0;padding:.33rem .65rem;line-height:1.45;transition:background .15s,border-color .15s}
nav.toc li.l2>a{color:#f8f9f5;font-weight:670;margin-top:.7rem}
nav.toc a:hover,nav.toc a.is-active{color:#fff;background:#27545b;border-left-color:#eb9477}
nav.toc a.is-active{font-weight:690}
.toc-search{display:block;width:100%;background:#244a53;color:#fff;border:1px solid #54717a;border-radius:5px;padding:.56rem .7rem;font:inherit;font-size:.79rem;margin-bottom:1rem}
.toc-search::placeholder{color:#bfd2d2}
.toc-empty{font-size:.78rem;color:#cadcdd;padding:.5rem}
main{padding:2.2rem 0 6rem}
main>p{line-height:1.78;color:#384c52}
h1{font-family:Georgia,"Times New Roman",serif;font-weight:700;font-size:clamp(2.05rem,3.1vw,2.9rem);line-height:1.19;letter-spacing:-.025em;max-width:58rem;margin:.2rem 0 1.3rem;color:#173943}
h1:before{content:"ИССЛЕДОВАНИЕ  /  LAB 01";display:block;font-family:"Segoe UI",Arial,sans-serif;color:#b15b42;font-weight:750;letter-spacing:.17em;font-size:.7rem;margin:0 0 1rem}
h2{font-family:Georgia,"Times New Roman",serif;color:#193c46;font-size:1.8rem;letter-spacing:-.02em;padding:1.5rem 0 .58rem;border-top:2px solid #193f48;border-bottom:1px solid #d7dfdc;margin:3.5rem 0 1.2rem}
h3{font-family:"Segoe UI",Arial,sans-serif;color:#1c4952;font-size:1.23rem;letter-spacing:-.01em;margin:2.5rem 0 .7rem;padding-left:.85rem;border-left:3px solid #d4785b}
h2 .num{color:#bd654e;font-family:"Segoe UI",Arial,sans-serif;font-size:.78em;font-weight:800;margin-right:.5em;font-variant-numeric:tabular-nums}
h3 .num{color:#a85a47;font-weight:700}
h2[id],h3[id]{scroll-margin-top:5.4rem}
.page-index h1{font-size:clamp(2.5rem,4vw,3.65rem);max-width:55rem;margin-bottom:1.6rem}
blockquote{border-left:4px solid #bc7557;background:#f8eee9;color:#4a514b;padding:.7rem 1.3rem;border-radius:0 5px 5px 0}
hr{border-color:#d3deda;margin:2.5rem 0}
nav.toc .num,details.toc-mobile .num{font-weight:inherit}
p{margin:.7em 0}
ul,ol{padding-left:1.4rem}
li{margin:.25em 0}
hr{border:0;border-top:1px solid var(--line);margin:2rem 0}
code{font:.88em/1.5 ui-monospace,"Cascadia Mono",Consolas,Menlo,monospace;background:var(--code-bg);padding:.08em .32em;border-radius:4px}
blockquote{margin:1em 0;padding:.2em 1em;border-left:3px solid var(--accent);background:var(--accent-bg);border-radius:0 6px 6px 0}
.table-wrap{overflow-x:auto;width:fit-content;max-width:100%;margin:.9rem 0 1.3rem;border:1px solid var(--line);border-radius:8px;background:var(--surface)}
.table-wrap{border-radius:7px;border:1px solid #b7cdca;border-top:3px solid #126e79;margin:1.2rem 0 2rem;background:#fff;box-shadow:0 8px 22px #24464d0b}
.table-wrap.md-table{width:auto}
table{border-collapse:collapse;font-size:.86rem;line-height:1.35;font-variant-numeric:tabular-nums}
.md-table table{width:100%}
table.rowhead td:first-child{font-weight:600;text-align:left;color:var(--ink)}
th,td{padding:.42rem .7rem;border-bottom:1px solid var(--line);text-align:right;vertical-align:top;white-space:nowrap}
thead th{background:#f6f5f2;color:var(--ink2);font-weight:600;border-bottom:1px solid var(--muted);vertical-align:bottom}
thead th{background:#e6f0ee;color:#173f46;border-bottom:2px solid #a9cbc6;font-weight:750}
tbody tr:nth-child(even){background:#f4f8f7}
tbody tr:hover{background:#eaf5f2}
th,td{padding:.68rem .88rem;line-height:1.45}
.md-table td:first-child,.md-table th:first-child{font-weight:650;color:#26434a}
tbody th{font-weight:600;text-align:left;color:var(--ink)}
tbody tr:last-child>*{border-bottom:0}
td.txt,th.txt{text-align:left;white-space:normal;min-width:8rem}
.md-table td,.md-table th{white-space:normal;text-align:left}
.out{margin:.6rem 0}
.out img{display:block;max-width:100%;height:auto;margin:0 auto;border:1px solid var(--line);border-radius:8px;background:#fff}
.figure-plate{max-width:var(--w-wide);margin:1.5rem 0 2rem;background:white;border:1px solid #d0dfdd;border-top:4px solid #397e89;border-radius:6px;overflow:hidden;box-shadow:0 11px 26px #153b460e}
.figure-plate .out{margin:0}
.figure-plate img{border:0;border-radius:0;box-shadow:none;cursor:zoom-in}
.figure-index{display:flex;align-items:center;gap:.9rem;margin:0;padding:.8rem 1.2rem;color:#1b5862;font-size:.7rem;font-weight:750;letter-spacing:.12em;text-transform:uppercase;background:#eff6f5}
.figure-index:after{content:"";height:1px;background:#bfd8d6;flex:1}
.figure-zoom{background:#fff;border:1px solid #91b5b5;color:#26626a;border-radius:5px;font:inherit;font-size:.68rem;letter-spacing:0;padding:.25rem .55rem;cursor:pointer}
.figure-zoom{margin-left:auto}
.figure-plate .out{padding:1rem 1.1rem .55rem}
.figure-plate p.caption{margin:0!important;padding:.85rem 1.25rem 1.1rem!important;border:0!important;border-top:1px solid #e1eae8!important;background:#f9fbfa;color:#51666a;font-size:.86rem;line-height:1.65}
.figure-key{display:flex;align-items:center;gap:1.25rem;flex-wrap:wrap;padding:.2rem 1.2rem .8rem;font-size:.73rem;color:#53696d}
.figure-key .swatch{display:inline-block;vertical-align:middle;width:.72rem;height:.72rem;border-radius:2px;margin-right:.35rem}
dialog.figure-modal{max-width:96vw;max-height:94vh;border:0;border-radius:6px;padding:1rem;background:#fff;box-shadow:0 30px 90px #0008}
dialog.figure-modal::backdrop{background:#0c222bcf;backdrop-filter:blur(3px)}
.figure-modal img{display:block;max-width:92vw;max-height:84vh;object-fit:contain}
.figure-modal button{display:block;margin:0 0 .7rem auto;background:#173942;border:0;border-radius:4px;padding:.4rem .8rem;color:#fff;cursor:pointer}
.out pre.stream{margin:.4rem 0;padding:.6rem .9rem;background:var(--code-bg);border-radius:6px;font:.82rem/1.5 ui-monospace,Consolas,monospace;white-space:pre-wrap;overflow-x:auto}
p.caption{font-size:.9rem;line-height:1.5;color:var(--ink2);margin:.5rem 0 1.6rem;padding-left:.9rem;border-left:3px solid var(--line);max-width:var(--w-wide)}
p.caption:not(.tab){padding:.6rem 0 0;border-left:0;border-top:1px solid #e1e5e0;margin:.35rem 0 2.1rem;font-size:.85rem;color:#5e6962;line-height:1.65}
p.caption.tab{padding-left:0;border-left:0;margin:1.6rem 0 .55rem;color:#31453a;font-size:.9rem}
p.caption strong:first-of-type{color:var(--ink)}
p.caption.tab{margin:1.4rem 0 .4rem}
details.code{margin:.3rem 0}
details.code>summary{list-style:none;cursor:pointer;display:inline-flex;align-items:center;gap:.4rem;font-size:.78rem;color:var(--muted);
padding:.12rem .55rem;border:1px solid var(--line);border-radius:999px;background:var(--surface);user-select:none}
details.code>summary::-webkit-details-marker{display:none}
details.code>summary::before{content:"";width:.42em;height:.42em;border-right:1.5px solid currentColor;border-bottom:1.5px solid currentColor;transform:rotate(-45deg);transition:transform .15s}
details.code[open]>summary::before{transform:rotate(45deg)}
details.code>summary:hover{color:var(--ink);border-color:var(--muted)}
details.code>summary .hide{display:none}
details.code[open]>summary .show{display:none}
details.code[open]>summary .hide{display:inline}
details.code .highlight{margin:.4rem 0 .8rem;background:var(--code-bg);border:1px solid var(--line);border-radius:8px;overflow-x:auto}
details.code pre{margin:0;padding:.8rem 1rem;font:.8rem/1.55 ui-monospace,"Cascadia Mono",Consolas,Menlo,monospace}
.back{font-size:.85rem;margin-top:1.6rem}
.back a{color:var(--muted);text-decoration:none}
.back a:hover{color:var(--accent)}
mjx-container[display=true]{overflow-x:auto;overflow-y:hidden;padding:.2rem 0;max-width:100%}
footer{border-top:1px solid var(--line);color:var(--muted);font-size:.85rem;padding:1.2rem;text-align:center}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(17rem,1fr));gap:1rem;max-width:var(--w-wide)!important;margin:1.4rem 0}
.card{display:block;padding:1.1rem 1.2rem;border:1px solid var(--line);border-radius:10px;background:var(--surface);text-decoration:none;color:var(--ink)}
.card{border:1px solid #cbdedb;border-top:4px solid #247c85;border-radius:8px;padding:1.6rem 1.5rem;box-shadow:0 9px 24px #153b460c;transition:transform .2s,box-shadow .2s}
.card:nth-child(2){border-top-color:#ce7657}
.card:hover{transform:translateY(-3px);box-shadow:0 16px 28px #153b461b;border-color:#5f9ea4}
.card .k{color:#b5634c;font-weight:750;letter-spacing:.14em}
.card h3{border:0;padding:0;color:#173a43;font-family:Georgia,"Times New Roman",serif;font-size:1.37rem}
.page-index .cards{margin:2rem 0 3rem}
.page-index main>h1{position:relative;background:#173942;color:#fff;max-width:var(--w-wide);border-bottom:5px solid #d27a5c;margin:0 0 1.5rem;padding:clamp(1.7rem,4vw,3.5rem);border-radius:8px 8px 0 0;isolation:isolate}
.page-index main>h1:before{color:#eab19c}
.page-index main>h1:after{content:"";position:absolute;right:1.6rem;bottom:1rem;width:6rem;height:6rem;background:radial-gradient(circle at center,#d9ebe6 1.8px,transparent 2.4px) 0 0 / 21px 21px;opacity:.3;z-index:-1}
.page-index main>h1+p{font-size:1.13rem;line-height:1.7;color:#34515a;max-width:47rem;margin:1.4rem 0 2rem}
footer{background:#183942;color:#d2e1df;border:0;padding:1.8rem}
footer a{color:#b9e6df}
.card:hover{border-color:var(--accent);box-shadow:0 2px 10px rgba(31,95,174,.08)}
.card .k{font-size:.72rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.card h3{margin:.25rem 0 .4rem;font-size:1.15rem}
.card p{margin:0;color:var(--ink2);font-size:.92rem;line-height:1.5}
.page-index main{margin:0 auto;max-width:var(--w-wide)}
.page-index .layout{display:block;max-width:64rem}
@media (max-width:62rem){
 .layout{display:block;padding:0 1rem}
 nav.toc{display:none}
 details.toc-mobile{display:block;margin:1rem 0;border:1px solid var(--line);border-radius:8px;background:var(--surface);padding:.5rem .9rem;font-size:.9rem}
 details.toc-mobile summary{cursor:pointer;font-weight:600}
 details.toc-mobile ol{list-style:none;padding:0;margin:.5rem 0 0}
 details.toc-mobile li.l3{padding-left:1rem}
 details.toc-mobile a{display:block;padding:.25rem 0;text-decoration:none}
 body{font-size:16px}
 h1{font-size:1.6rem} h2{font-size:1.3rem}
}
@media print{.top,nav.toc,details.toc-mobile,details.code,.back{display:none}.layout{display:block}body{font-size:11pt;background:#fff}}
@media(max-width:62rem){
 .layout{padding:0 1rem}.top{gap:.5rem 1rem}.top .brand{width:100%;margin-bottom:.15rem}
 details.toc-mobile{background:#e6f0ee;border:1px solid #aacbc6;padding:.75rem 1rem}
 h1{font-size:2rem}h2{font-size:1.53rem}h3{font-size:1.14rem}
 .figure-index{padding:.65rem .8rem}.figure-plate .out{padding:.5rem}
}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}.card,nav.toc a{transition:none}}
@media print{.figure-plate{box-shadow:none;break-inside:avoid}footer{background:white;color:#333}.figure-zoom{display:none}}
'''

JS = r'''
document.getElementById('toggle-code')?.addEventListener('click', function () {
  var all = document.querySelectorAll('details.code');
  var open = !this.dataset.open;
  all.forEach(function (d) { d.open = open; });
  this.dataset.open = open ? '1' : '';
  this.textContent = open ? 'Скрыть весь код' : 'Показать весь код';
});
// Subtle reading-state indicator: the one visible contents menu follows the article.
var tocLinks = Array.from(document.querySelectorAll('nav.toc a[data-anchor]'));
var tocHeads = tocLinks.map(function (link) { return document.getElementById(link.dataset.anchor); });
var pendingTocUpdate = false;
function updateToc() {
  var active = -1;
  tocHeads.forEach(function (head, index) {
    if (head && head.getBoundingClientRect().top < 170) active = index;
  });
  tocLinks.forEach(function (link, index) {
    link.classList.toggle('is-active', active === index);
    if (active === index) link.setAttribute('aria-current', 'location');
    else link.removeAttribute('aria-current');
  });
  pendingTocUpdate = false;
}
window.addEventListener('scroll', function () {
  if (!pendingTocUpdate) {
    pendingTocUpdate = true;
    requestAnimationFrame(updateToc);
  }
}, {passive:true});
updateToc();
document.getElementById('toc-search')?.addEventListener('input', function () {
  var query = this.value.trim().toLocaleLowerCase('ru');
  var matches = 0;
  document.querySelectorAll('nav.toc li').forEach(function (li) {
    var visible = li.textContent.toLocaleLowerCase('ru').includes(query);
    li.hidden = !visible;
    if (visible) matches++;
  });
  var empty = document.querySelector('.toc-empty');
  if (empty) empty.hidden = matches > 0;
});
document.addEventListener('click', function (event) {
  var control = event.target.closest('.figure-zoom, .figure-plate img');
  if (!control) return;
  var plate = control.closest('.figure-plate');
  var original = plate && plate.querySelector('img');
  var dialog = document.getElementById('figure-modal');
  if (!original || !dialog || !dialog.showModal) return;
  var large = dialog.querySelector('img');
  large.src = original.src;
  large.alt = original.alt;
  dialog.showModal();
});
document.querySelector('#figure-modal button')?.addEventListener('click', function () {
  document.getElementById('figure-modal').close();
});
document.getElementById('figure-modal')?.addEventListener('click', function (event) {
  if (event.target === this) this.close();
});
// Переход по якорю внутри страницы. Нужен там, где страница подставляется скриптом
// (htmlpreview.github.io) и браузер сам к якорю не прокручивает.
function goTo(id) {
  var el = document.getElementById(id);
  if (el) { el.scrollIntoView(); }
  return !!el;
}
document.addEventListener('click', function (e) {
  var a = e.target.closest ? e.target.closest('a[data-anchor]') : null;
  if (!a || !goTo(a.dataset.anchor)) { return; }
  e.preventDefault();
  try { history.pushState(null, '', '#' + a.dataset.anchor); } catch (err) {}
});
window.addEventListener('popstate', function () { goTo(decodeURIComponent(location.hash.slice(1))); });
// Формулы: если MathJax загрузился, а страница не размечена, запускаем разметку сами,
// затем возвращаемся к якорю из адреса (высота страницы после разметки меняется).
(function retry(n) {
  var done = document.querySelector('mjx-container');
  if (!done && window.MathJax && MathJax.typesetPromise) {
    MathJax.typesetPromise().then(function () { goTo(decodeURIComponent(location.hash.slice(1))); });
    return;
  }
  if (done) { goTo(decodeURIComponent(location.hash.slice(1))); return; }
  if (n > 0) { setTimeout(function () { retry(n - 1); }, 600); }
})(25);
'''

NUMERIC = re.compile(r'^[\s−\-+]?(\d[\d\s.,]*)(·10[⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+|e[+-]?\d+)?%?$|^[—∞0]$|^−∞$')


def png_size(b64):
    raw = base64.b64decode(b64[:64])
    return struct.unpack('>II', raw[16:24])


def render_output(out):
    kind = out['output_type']
    if kind == 'stream':
        text = ''.join(out['text']).rstrip()
        return f'<div class="out"><pre class="stream">{html.escape(text)}</pre></div>' if text else ''
    if kind in ('display_data', 'execute_result'):
        data = out['data']
        if 'image/png' in data:
            b64 = ''.join(data['image/png']).replace('\n', '')
            width, height = (round(v * 96 / FIG_DPI) for v in png_size(b64))
            return (f'<div class="out wide"><img alt="" width="{width}" height="{height}" '
                    f'src="data:image/png;base64,{b64}"></div>')
        if 'text/markdown' in data:
            return markdown2html_mistune(''.join(data['text/markdown']))
        if 'text/html' in data:
            return f'<div class="out">{"".join(data["text/html"])}</div>'
        if 'text/plain' in data:
            return f'<div class="out"><pre class="stream">{html.escape("".join(data["text/plain"]))}</pre></div>'
    if kind == 'error':
        raise RuntimeError(f'В ноутбуке сохранена ошибка выполнения: {out.get("ename")}')
    return ''


def render_cells(nb):
    formatter = HtmlFormatter(nowrap=False, cssclass='highlight')
    parts = []
    for cell in nb.cells:
        src = cell.source
        if not src.strip():
            continue
        if cell.cell_type == 'markdown':
            parts.append(markdown2html_mistune(src))
        elif cell.cell_type == 'code':
            code = highlight(src, PythonLexer(), formatter)
            parts.append('<details class="code"><summary><span class="show">Показать код</span>'
                         f'<span class="hide">Скрыть код</span></summary>{code}</details>')
            parts.extend(render_output(o) for o in cell.get('outputs', []))
    return '\n'.join(parts)


def polish(body, page):
    """Привести разметку в порядок: якоря, оглавление, подписи, таблицы, ссылки между страницами."""
    soup = BeautifulSoup(body, 'html.parser')

    for a in soup.select('a.anchor-link'):
        a.decompose()

    # якорь, стоящий перед заголовком, становится id самого заголовка
    toc = []
    for h in soup.find_all(['h1', 'h2', 'h3', 'h4']):
        prev = h.find_previous_sibling()
        anchor = None
        if prev is not None:
            if prev.name == 'a' and prev.get('id') and not prev.get_text(strip=True):
                anchor = prev
            elif prev.name == 'p' and not prev.get_text(strip=True):
                found = prev.find_all('a', id=True)
                if len(found) == 1:
                    anchor = prev
        if anchor is not None:
            ident = anchor.get('id') or anchor.find('a', id=True)['id']
            anchor.decompose()
            h['id'] = ident
        text = h.get_text(' ', strip=True)
        m = re.match(r'^(\d+(?:\.\d+)*\.)\s+(.*)$', text)
        if m and h.contents and isinstance(h.contents[0], str) and h.contents[0].lstrip().startswith(m.group(1)):
            first = h.contents[0]
            rest = first.lstrip()[len(m.group(1)):].lstrip()
            num = soup.new_tag('span', attrs={'class': 'num'}); num.string = m.group(1)
            first.replace_with(num, rest)
        if h.name in ('h2', 'h3') and h.get('id'):
            toc.append((h.name, h['id'], h.decode_contents()))

    # В ноутбуке сохранено собственное оглавление для Jupyter. В HTML меню уже
    # вынесено в боковую колонку: убираем копию, не трогая исходный ноутбук.
    for h in soup.find_all('h2', id='toc'):
        sibling = h.find_next_sibling()
        while sibling is not None and sibling.name not in ('h1', 'h2'):
            following = sibling.find_next_sibling()
            sibling.decompose()
            sibling = following
        h.decompose()
    toc = [item for item in toc if item[1] != 'toc']

    # подписи к рисункам и таблицам
    for a in soup.find_all('a', id=re.compile(r'^(fig|tab)-')):
        p = a.find_parent('p')
        if p is not None:
            p['class'] = ['caption'] + (['tab'] if a['id'].startswith('tab-') else [])
    for p in soup.find_all('p'):
        if p.find('a', href='#toc') and len(p.get_text(strip=True)) < 40:
            p['class'] = ['back']

    # альтернативный текст рисунка берём из подписи под ним
    for img in soup.find_all('img'):
        cap = img.find_parent('div').find_next_sibling('p', class_='caption')
        if cap is not None:
            img['alt'] = re.sub(r'\s+', ' ', cap.get_text(' ', strip=True))[:300]
            anchor = cap.find('a', id=re.compile(r'^fig-'))
            parent = img.find_parent('div', class_='out')
            if anchor is not None and parent is not None:
                plate = soup.new_tag('figure', attrs={'class': 'figure-plate'})
                parent.wrap(plate)
                label = soup.new_tag('div', attrs={'class': 'figure-index'})
                figure_topics = {
                    'sprint1.html': {
                        'fig-0-1': 'Дизайн исследования',
                        'fig-1-1': 'Разведочный анализ', 'fig-1-2': 'Подбор распределения',
                        'fig-1-3': 'Связь величин', 'fig-1-4': 'Состоятельность оценок',
                        'fig-1-5': 'Состоятельность оценок', 'fig-2-1': 'Проверка предсказаний',
                        'fig-2-2': 'Разведочный анализ', 'fig-2-3': 'Подбор распределения',
                        'fig-2-4': 'Состоятельность оценок', 'fig-2-5': 'Состоятельность оценок'},
                    'sprint2.html': {
                        'fig-0-1': 'План исследования', 'fig-1-1': 'Проверка генератора',
                        'fig-2-1': 'Проверка генератора', 'fig-3-1': 'Проверка генератора',
                        'fig-4-1': 'Проверка генератора', 'fig-5-1': 'Сравнение с данными',
                        'fig-5-2': 'Состоятельность оценок'}
                }
                k = anchor['id']
                label.string = figure_topics.get(page, {}).get(k, 'Иллюстрация')
                button = soup.new_tag('button', attrs={'class': 'figure-zoom', 'type': 'button',
                                                       'aria-label': 'Увеличить рисунок'})
                button.string = 'Развернуть ↗'
                label.append(button)
                plate.insert(0, label)
                # Пояснение остаётся оригинальным: подпись из ноутбука переносим
                # непосредственно под график как журнальную подпись.
                plate.append(cap.extract())
                key = soup.new_tag('div', attrs={'class': 'figure-key'})
                roles = [('Данные', '#2a78d6')]
                if page == 'sprint2.html' and k in {'fig-1-1', 'fig-2-1', 'fig-3-1', 'fig-4-1'}:
                    roles = [('Генератор', '#1baf7a'), ('Модель', '#eb6834')]
                elif page == 'sprint2.html' and k == 'fig-5-1':
                    roles = [('Данные', '#2a78d6'), ('Генератор', '#1baf7a')]
                elif page == 'sprint2.html' and k == 'fig-0-1':
                    roles = [('Данные', '#2a78d6'), ('Модель', '#eb6834'), ('Генератор', '#1baf7a')]
                elif page == 'sprint1.html' and k in {'fig-1-2', 'fig-2-3'}:
                    roles.append(('Модель', '#eb6834'))
                elif page == 'sprint1.html' and k == 'fig-0-1':
                    roles = [('Данные', '#2a78d6'), ('Модель', '#eb6834'), ('Генератор', '#1baf7a')]
                # Не показывать ложную универсальную легенду на диаграммах
                # состоятельности и паритетных графиках.
                if k not in {'fig-1-4', 'fig-1-5', 'fig-2-1', 'fig-2-4', 'fig-2-5', 'fig-5-2'}:
                    for name, color in roles:
                        dot = soup.new_tag('span', attrs={'class': 'swatch', 'style': f'background:{color}'})
                        item = soup.new_tag('span')
                        item.append(dot)
                        item.append(name)
                        key.append(item)
                    plate.insert(2, key)

    # таблицы
    for style in soup.find_all('style'):
        style.decompose()
    for t in soup.find_all('table'):
        is_df = 'dataframe' in (t.get('class') or [])
        for attr in ('border', 'style'):
            t.attrs.pop(attr, None)
        for tr in t.find_all('tr'):
            tr.attrs.pop('style', None)
        if is_df:
            for cell in t.find_all(['td', 'th']):
                text = cell.get_text(strip=True)
                if cell.name == 'td' and text and not NUMERIC.match(text):
                    cell['class'] = ['txt']
        wrap = soup.new_tag('div', attrs={'class': 'table-wrap' + ('' if is_df else ' md-table')})
        holder = t.parent if (is_df and t.parent.name == 'div' and t.parent.get('class') is None) else t
        holder.wrap(wrap)
    for out in soup.select('div.out'):
        inner = out.find('div', class_='table-wrap')
        if inner is not None:
            out.replace_with(inner)

    # исходный код функций из разделов «code»: подсветка и тот же вид, что у свёрнутых ячеек
    for det in soup.select('details.src'):
        pre = det.find('pre')
        code = highlight(pre.get_text(), PythonLexer(), HtmlFormatter(cssclass='highlight'))
        pre.replace_with(BeautifulSoup(code, 'html.parser'))
        det['class'] = ['code', 'src']
        box = det.find_parent('div', class_='out')
        if box is not None:
            box.replace_with(det)

    # ссылки на соседний ноутбук ведут на соседнюю страницу
    # приглашение перейти в HTML-версию нужно только в ноутбуке
    for quote in soup.find_all('blockquote'):
        if quote.find('a', href=re.compile('htmlpreview')):
            quote.decompose()

    for a in soup.find_all('a', href=True):
        href = a['href']
        if href.startswith('#'):
            a['data-anchor'] = href[1:]
        for nb_name, target in LINKS.items():
            if href.startswith(nb_name):
                a['href'] = target + href[len(nb_name):]
        if a['href'].startswith('http'):
            a['rel'] = 'noopener'

    return str(soup), toc


def toc_html(toc):
    items = '\n'.join(f'<li class="l{name[1]}"><a href="#{ident}" data-anchor="{ident}">{label}</a></li>' for name, ident, label in toc)
    return f'<ol>{items}</ol>'


def shell(title, body, toc, current, extra_class=''):
    nav = ''.join(f'<a class="nav" href="{page}"{" aria-current=page" if page == current else ""}>{label}</a>'
                  for _, page, label in PAGES)
    home = ' aria-current=page' if current == 'index.html' else ''
    side = (f'<nav class="toc" aria-label="Оглавление"><p class="toc-title">Разделы исследования</p>'
            f'<input id="toc-search" class="toc-search" type="search" placeholder="Найти раздел" '
            f'aria-label="Поиск по оглавлению">{toc_html(toc)}'
            f'<p class="toc-empty" hidden>Раздел не найден</p></nav>'
            if toc else '')
    mobile = f'<details class="toc-mobile"><summary>Оглавление</summary>{toc_html(toc)}</details>' if toc else ''
    button = '<button id="toggle-code" type="button">Показать весь код</button>' if toc else ''
    pygments_css = HtmlFormatter(style='default').get_style_defs('.highlight')
    page_class = current.removesuffix('.html')
    return f'''<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{CSS}
{pygments_css}
.highlight{{background:var(--code-bg)}}
</style>
<script>
window.MathJax = window.MathJax || {{
  tex: {{inlineMath: [['$', '$'], ['\\\\(', '\\\\)']], displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']], processEscapes: true}},
  svg: {{fontCache: 'global'}}
}};
// MathJax подключается из скрипта, а не тегом: так он загружается и там,
// где страницу подставляет сторонний скрипт (htmlpreview.github.io).
if (!document.getElementById('mathjax-loader')) {{
  var loader = document.createElement('script');
  loader.id = 'mathjax-loader'; loader.async = true; loader.src = '{MATHJAX}';
  document.head.appendChild(loader);
}}
</script>
</head>
<body class="{extra_class} page-{page_class}">
<header class="top">
<a class="brand" href="index.html"><span class="brand-symbol">01</span> ИнфоХимия / Статистика</a>
<a class="nav" href="index.html"{home}>Обзор</a>{nav}
{button}
</header>
<div class="layout">
{side}
<main>
<div id="toc"></div>
{mobile}
{body}
</main>
</div>
<dialog id="figure-modal" class="figure-modal" aria-label="Просмотр графика крупно"><button type="button">Закрыть ×</button><img alt=""></dialog>
<footer>Собрано из ноутбуков репозитория <a href="{REPO_URL}">statistics_lab_01</a>. Код свёрнут и раскрывается по клику.</footer>
<script>{JS}</script>
</body>
</html>
'''


def build_page(nb_name, page):
    nb = nbformat.read(ROOT / nb_name, as_version=4)
    body, toc = polish(render_cells(nb), page)
    title = next((c.source for c in nb.cells if c.cell_type == 'markdown' and '# ' in c.source), nb_name)
    title = re.sub(r'[#$\\{}^_]', '', re.search(r'^# (.*)$', title, re.M).group(1)).strip()
    (DOCS / page).write_text(shell(title, body, toc, page), encoding='utf-8', newline='\n')
    n_code = sum(1 for c in nb.cells if c.cell_type == 'code' and c.source.strip())
    print(f'{page}: {len(toc)} пунктов оглавления, {n_code} свёрнутых ячеек кода, '
          f'{(DOCS / page).stat().st_size / 1e6:.1f} МБ')


def build_index():
    src = (ROOT / 'tools' / 'index.md').read_text(encoding='utf-8')
    body, _ = polish(markdown2html_mistune(src), 'index.html')
    (DOCS / 'index.html').write_text(shell('Лабораторная работа 1. Обзор', body, [], 'index.html', 'page-index'),
                                     encoding='utf-8', newline='\n')
    print('index.html')


if __name__ == '__main__':
    DOCS.mkdir(exist_ok=True)
    (DOCS / '.nojekyll').write_text('', encoding='utf-8')
    for nb_name, page, _ in PAGES:
        build_page(nb_name, page)
    build_index()
