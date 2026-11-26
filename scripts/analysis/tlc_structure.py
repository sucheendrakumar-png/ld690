"""Line structure and signature words on TLC, with the same definitions as
scripts/songmap/structure.py and trends.py (FINGER words), for the seven LD-690 poets that
TLC has (Bharathi is not in TLC). Overlapping songs removed. Writes out/tlc_structure.json.
Caveat: TLC line breaks often split a sung line in two, which changes line-pair measures."""
import json, os, re, sys, collections
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'paper', 'analysis'))
import csv; csv.field_size_limit(10 ** 9)
import robustness as rb
G = re.compile(r'[அ-ஹஃ][ா-்ௗ]*'); VS = re.compile(r'[ா-்ௗ]')
def letters(l): return G.findall(l)
def base(g): return VS.sub('', g)
rows = list(csv.DictReader(open(os.path.join(ROOT, 'paper', 'ext', 'tlc', 'Tamil_lyrics_v1.csv'), encoding='utf-8')))
keep = {i for i, t in enumerate(rb.TLC) if t not in [rb.TLC[j] for j in rb.over_t]} if False else None
# rebuild TLC with raw text, aligned to rb.TLC filtering (>=10 Tamil tokens), then drop overlap
raw = [r for r in rows if len(rb.words(r['lyrics_tamil'])) >= 10]
assert len(raw) == len(rb.TLC)
raw = [r for j, r in enumerate(raw) if j not in rb.over_t]
POETS = ['Vairamuthu', 'Kannadasan', 'Vaali', 'Na. Muthukumar', 'Thamarai', 'Madhan Karky']
songs = collections.defaultdict(list)
for r in raw:
    p = rb.norm(r['lyricist'])
    seen, lines = set(), []
    for l in r['lyrics_tamil'].split('\n'):
        l = l.strip()
        if l and l not in seen and len(letters(l)) >= 4: seen.add(l); lines.append(l)
    if len(lines) >= 6: songs[p].append(lines)
R = np.random.default_rng(7)
def feats(lines):
    L = [letters(l) for l in lines]; pr = list(zip(L, L[1:]))
    n = np.array([len(x) for x in L])
    return dict(ethugai=np.mean([a[1] == b[1] for a, b in pr]), monai=np.mean([base(a[0]) == base(b[0]) for a, b in pr]),
                iyaibu=np.mean([a[-1] == b[-1] for a, b in pr]), linecv=n.std() / n.mean())
words_by = {p: [w for ls in v for l in ls for w in re.findall(r'[஀-௿]+', l)] for p, v in songs.items()}
owner = collections.defaultdict(set)
for p, ws in words_by.items():
    for w in set(ws): owner[w].add(p)
out = {'structure': [], 'finger': []}
for p in POETS:
    F = [feats(ls) for ls in songs[p]]; row = {'poet': p, 'n': len(F)}
    for k in F[0]:
        v = np.array([f[k] for f in F]); bs = [v[R.integers(0, len(v), len(v))].mean() for _ in range(2000)]
        row[k] = [round(float(v.mean()), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]
    ws = words_by[p]
    cs = [np.mean([len(owner[w]) == 1 for w in set(R.choice(ws, 1500, replace=False))]) for _ in range(200)]
    row['coined'] = [round(float(np.mean(cs)), 3), round(float(np.percentile(cs, 2.5)), 3), round(float(np.percentile(cs, 97.5)), 3)]
    out['structure'].append(row)
FINGER = {'Kannadasan': ['இறைவன்', 'மனிதன்', 'ஆசை'], 'Vairamuthu': ['கண்ணீர்', 'மண்ணில்', 'பூமி'],
          'Vaali': ['அன்பு', 'தெய்வம்', 'தாய்'], 'Na. Muthukumar': ['பெண்ணே', 'அடடா', 'நொடி'],
          'Thamarai': ['கொஞ்சம்', 'ஏதோ', 'ஏனோ'], 'Madhan Karky': ['எந்தன்', 'உந்தன்', 'உயிரே']}
T = [t for j, t in enumerate(rb.TLC) if j not in rb.over_t]
for t in T: t['ws'] = set(t['w'])
for p, ws in FINGER.items():
    mine = [t for t in T if t['lyr'] == p]; rest = [t for t in T if t['lyr'] != p]
    out['finger'].append({'poet': p, 'n': len(mine), 'words': [
        {'w': w, 'a': round(100 * sum(w in t['ws'] for t in mine) / len(mine), 1),
         'b': round(100 * sum(w in t['ws'] for t in rest) / len(rest), 1)} for w in ws]})
json.dump(out, open(os.path.join(rb.OUT, 'tlc_structure.json'), 'w'), ensure_ascii=False, indent=1)
for r in out['structure']:
    print(f"{r['poet']:<16}{r['n']:>5} " + ' '.join(f"{k} {100*r[k][0]:.0f}[{100*r[k][1]:.0f}-{100*r[k][2]:.0f}]" for k in ['ethugai', 'monai', 'iyaibu', 'linecv', 'coined']))
for f in out['finger']: print(f['poet'], f['n'], [(w['w'], w['a'], w['b']) for w in f['words']])
