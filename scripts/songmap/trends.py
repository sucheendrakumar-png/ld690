"""How the words changed: era trends in the Tamil of the site's film songs, and which of
them survive the site's own song selection. Writes data/songmap/trends.json for build_data.py.

A finding is kept on the page only if it holds within love songs alone, within philosophy
songs alone and within single poets' own careers, so that it is not just an artefact of
which songs the site chose to translate (see bias_check.py)."""
import json, re, ast, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SM = os.path.join(ROOT, 'data', 'songmap')
S = [s for s in json.load(open(os.path.join(SM, 'songs.json'))) if s['english'] and s['tamil'].strip()]  # the 689 parallel songs
LYR = {'Vasana Kavidhai': 'Subramanya Bharathi', 'Na.Muthukumar': 'Na. Muthukumar'}
ERAS = [(1950, 1979), (1980, 1999), (2000, 2030)]
TA = re.compile(r'[஀-௿]+')

for s in S:
    s['lyr'] = LYR.get(s['lyricist'], s['lyricist'])
    try: s['y'] = int(str(s['year'])[:4])
    except Exception: s['y'] = None
    m = s['moods']; s['m'] = ast.literal_eval(m) if isinstance(m, str) else m
    seen, lines = set(), []
    for l in s['tamil'].split('\n'):
        l = l.strip()
        if l and l not in seen: seen.add(l); lines.append(l)
    s['w'] = TA.findall(' '.join(lines))          # refrains counted once
    s['ws'] = set(s['w'])

film = [s for s in S if s['y'] and s['y'] >= 1950 and s['w']
        and 'Bharathi' not in s['lyr'] and 'Siddhar' not in s['lyr']]

YOU = {'நீ', 'நீதானே', 'நீதான்', 'நீதானம்மா', 'நீங்க', 'நீங்கள்', 'நீவிர்'}
def is_you(x): return x in YOU or x.startswith('நீய')   # not நீர் (water), நீதி, நீல, நீங்கு (leave)…
def has(s, stem):
    if stem == 'நீ': return any(is_you(x) for x in s['w'])
    return any(x.startswith(stem) for x in s['w'])
def share(ss, stem): return round(100 * sum(has(s, stem) for s in ss) / len(ss)) if ss else None
def wlen(ss): return round(float(np.mean([len(w) for s in ss for w in s['w']])), 2) if ss else None
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(100 * (c - h)), round(100 * (c + h))
RNG = np.random.default_rng(7)
def ci(ss, key, stem):
    # 95% interval: Wilson for shares, bootstrap over songs for word length
    if key == 'len':
        L = [np.array([len(w) for w in s['w']]) for s in ss]
        bs = [np.concatenate([L[i] for i in RNG.integers(0, len(L), len(L))]).mean() for _ in range(1000)]
        return [round(float(np.percentile(bs, 2.5)), 2), round(float(np.percentile(bs, 97.5)), 2)]
    return list(wilson(sum(has(s, stem) for s in ss), len(ss)))
def by_era(f, val): return [{'v': val([s for s in film if a <= s['y'] <= b and f(s)]),
                             'n': sum(1 for s in film if a <= s['y'] <= b and f(s))} for a, b in ERAS]
def career(name, val):
    ss = sorted([s for s in film if s['lyr'] == name], key=lambda s: s['y'])
    h = len(ss) // 2; A, B = ss[:h], ss[h:]
    return {'poet': name, 'from': [A[0]['y'], A[-1]['y']], 'to': [B[0]['y'], B[-1]['y']],
            'a': val(A), 'b': val(B), 'n': len(ss)}

POETS = ['Kannadasan', 'Vaali', 'Vairamuthu']
TRENDS = [('nee', 'நீ', lambda ss: share(ss, 'நீ')),
          ('len', None, wlen),
          ('kaadhal', 'காதல்', lambda ss: share(ss, 'காதல்')),
          ('mazhai', 'மழை', lambda ss: share(ss, 'மழை'))]
out = {'eras': [[a, min(b, 2026)] for a, b in ERAS],
       'n': [sum(1 for s in film if a <= s['y'] <= b) for a, b in ERAS], 'trends': []}
for key, word, val in TRENDS:
    all_ = by_era(lambda s: True, val)
    for e, (a, b) in zip(all_, ERAS):
        e['ci'] = ci([s for s in film if a <= s['y'] <= b], key, word)
    out['trends'].append({'key': key, 'word': word,
        'all': all_,
        'love': by_era(lambda s: 'love' in s['m'], val),
        'phil': by_era(lambda s: 'philosophy' in s['m'], val),
        'noK': by_era(lambda s: s['lyr'] != 'Kannadasan', val),
        'careers': [career(p, val) for p in POETS]})


# each poet's fingerprint words: share of the poet's songs using the exact word vs everyone else's
FINGER = {'Kannadasan': ['இறைவன்', 'மனிதன்', 'ஆசை'], 'Vairamuthu': ['கண்ணீர்', 'மண்ணில்', 'பூமி'],
          'Vaali': ['அன்பு', 'தெய்வம்', 'தாய்'], 'Na. Muthukumar': ['பெண்ணே', 'அடடா', 'நொடி'],
          'Thamarai': ['கொஞ்சம்', 'ஏதோ', 'ஏனோ'], 'Madhan Karky': ['எந்தன்', 'உந்தன்', 'உயிரே'],
          'Subramanya Bharathi': ['ஞாயிறு', 'வாழ்க', 'நின்']}
out['finger'] = []
for p, words in FINGER.items():
    mine = [s for s in S if s['lyr'] == p and s['w']]; rest = [s for s in S if s['lyr'] != p and s['w']]
    out['finger'].append({'poet': p, 'n': len(mine), 'words': [
        {'w': w, 'a': round(100 * sum(w in s['ws'] for s in mine) / len(mine)),
         'b': round(100 * sum(w in s['ws'] for s in rest) / len(rest), 1)} for w in words]})

json.dump(out, open(os.path.join(SM, 'trends.json'), 'w'), ensure_ascii=False, indent=1)
for t in out['trends']:
    print(t['key'], [e['v'] for e in t['all']], 'love', [e['v'] for e in t['love']], 'phil', [e['v'] for e in t['phil']],
          [(c['poet'], c['a'], c['b']) for c in t['careers']])
for f in out['finger']: print(f['poet'], [(w['w'], w['a'], w['b']) for w in f['words']])
print(out['n'], [[e['ci'] for e in t['all']] for t in out['trends']])
