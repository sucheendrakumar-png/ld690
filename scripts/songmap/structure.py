"""Structural craft per poet: does any poet's line-building stand out?
Per song, over consecutive line pairs (refrains counted once, in order):
  ethugai  - 2nd letter of the two lines matches (classical Tamil rhyme on the second syllable)
  monai    - 1st letter of the two lines shares its base letter (alliteration)
  iyaibu   - last letter of the two lines matches (end rhyme)
  linecv   - spread of line lengths (std/mean of letters per line): lower = more even, metred lines
  coined   - share of the poet's distinct words found in no other poet's songs (equal-size samples)
  grantha  - ஜ ஷ ஸ ஹ ஶ per 1000 letters (loanword proxy)
Poets with 20+ songs. 95% bootstrap intervals over songs. Writes structure.json."""
import json, os, re, collections
import numpy as np
SM = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'songmap')
S = [s for s in json.load(open(os.path.join(SM, 'songs.json'))) if s['english'] and s['tamil'].strip()]  # the 689 parallel songs
LYR = {'Vasana Kavidhai': 'Subramanya Bharathi', 'Na.Muthukumar': 'Na. Muthukumar'}
G = re.compile(r'[அ-ஹஃ][ா-்ௗ]*')   # one Tamil letter (base + signs)
VS = re.compile(r'[ா-்ௗ]')
def letters(l): return G.findall(l)
def base(g): return VS.sub('', g)
rng = np.random.default_rng(7)
songs = collections.defaultdict(list)
for s in S:
    lyr = LYR.get(s['lyricist'], s['lyricist'])
    seen, lines = set(), []
    for l in s['tamil'].split('\n'):
        l = l.strip()
        if l and l not in seen and len(letters(l)) >= 4: seen.add(l); lines.append(l)
    if len(lines) >= 6: songs[lyr].append(lines)
poets = [p for p, v in songs.items() if len(v) >= 20]
def feats(lines):
    L = [letters(l) for l in lines]
    pr = list(zip(L, L[1:]))
    eth = np.mean([a[1] == b[1] for a, b in pr])
    mon = np.mean([base(a[0]) == base(b[0]) for a, b in pr])
    iya = np.mean([a[-1] == b[-1] for a, b in pr])
    n = np.array([len(x) for x in L]); cv = n.std() / n.mean()
    txt = ''.join(lines); gr = 1000 * sum(c in 'ஜஷஸஹஶ' for c in txt) / max(1, len(G.findall(txt)))
    return dict(ethugai=eth, monai=mon, iyaibu=iya, linecv=cv, grantha=gr)
words_by = {p: [w for ls in songs[p] for l in ls for w in re.findall(r'[஀-௿]+', l)] for p in songs}
owner = collections.defaultdict(set)
for p, ws in words_by.items():
    for w in set(ws): owner[w].add(p)
out = {'poets': []}
for p in sorted(poets, key=lambda p: -len(songs[p])):
    F = [feats(ls) for ls in songs[p]]
    row = {'poet': p, 'n': len(F)}
    for k in F[0]:
        v = np.array([f[k] for f in F]); bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(2000)]
        row[k] = [round(float(v.mean()), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]
    ws = words_by[p]
    cs = [np.mean([len(owner[w]) == 1 for w in set(rng.choice(ws, 1500, replace=False))]) for _ in range(200)]
    row['coined'] = [round(float(np.mean(cs)), 3), round(float(np.percentile(cs, 2.5)), 3), round(float(np.percentile(cs, 97.5)), 3)]
    out['poets'].append(row)
json.dump(out, open(os.path.join(SM, 'structure.json'), 'w'), ensure_ascii=False, indent=1)
for r in out['poets']:
    print(f"{r['poet']:<20}{r['n']:>4} " + ' '.join(f"{k} {r[k][0]:.3f}[{r[k][1]:.2f}-{r[k][2]:.2f}]" for k in ['ethugai', 'monai', 'iyaibu', 'linecv', 'coined', 'grantha']))
