"""Robustness checks on an independent corpus: the Tamil Lyrics Corpus of Chinnappa &
Dhandapani (2021), 5,449 songs scraped from allnewlyrics.com.
Get it once (not redistributed here, no licence stated):
    git clone --depth 1 https://github.com/praveenraj0904/tamillyricscorpus paper/ext/tlc
Re-runs, with the exact definitions used on LD-690 (scripts/songmap/trends.py and
lyricist_style.py):
  (1) share of songs using the second-person pronoun நீ, by era, with Wilson 95% CIs;
  (2) mean Tamil word length (code points), by era, bootstrap 95% CIs over songs;
  (3) both within single poets' careers (first vs second half of their dated songs);
  (4) the same 7-way char 1-3-gram LinearSVC lyricist identification, 10-fold CV, on
      LD-690 and TLC with the six film lyricists both corpora share (Bharathi is not in TLC),
      on TLC subsampled to LD-690's class sizes, and trained on TLC / tested on LD-690.
Writes paper/analysis/out/robustness.json."""
import csv, json, os, re, sys, collections
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'songmap'))
from tamil_style import tamil_doc
OUT = os.path.join(ROOT, 'paper', 'analysis', 'out'); os.makedirs(OUT, exist_ok=True)
csv.field_size_limit(10 ** 9)
TA = re.compile(r'[஀-௿]+')
ERAS = [(1950, 1979), (1980, 1999), (2000, 2030)]
YOU = {'நீ', 'நீதானே', 'நீதான்', 'நீதானம்மா', 'நீங்க', 'நீங்கள்', 'நீவிர்'}
def is_you(x): return x in YOU or x.startswith('நீய')
NORM = {'Vasana Kavidhai': 'Subramanya Bharathi', 'Na.Muthukumar': 'Na. Muthukumar', 'Na. Muthukumar': 'Na. Muthukumar',
        'Muthukumar': 'Na. Muthukumar', 'Na Muthukumar': 'Na. Muthukumar', 'Madhan karky': 'Madhan Karky', 'Na. Muthu Kumar': 'Na. Muthukumar'}
def norm(l): l = (l or '').strip(); return NORM.get(l, l)
def words(t):
    seen, ws = set(), []
    for l in t.split('\n'):
        l = l.strip()
        if l and l not in seen: seen.add(l); ws += TA.findall(l)   # refrains counted once
    return ws
def rec(lyr, year, tamil, src):
    try: y = int(str(year)[:4])
    except Exception: y = None
    w = words(tamil)
    return {'lyr': norm(lyr), 'y': y, 'w': w, 'doc': tamil_doc(tamil), 'src': src,
            'key': ' '.join(w[:12])}
LD = [rec(s['lyricist'], s['year'], s['tamil'], 'LD') for s in json.load(open(os.path.join(ROOT, 'data', 'songmap', 'songs.json')))
      if s['english'] and s['tamil'].strip()]   # the 689 parallel songs
TLC = [rec(r['lyricist'], r['year'], r['lyrics_tamil'], 'TLC')
       for r in csv.DictReader(open(os.path.join(ROOT, 'paper', 'ext', 'tlc', 'Tamil_lyrics_v1.csv'), encoding='utf-8'))]
LD = [s for s in LD if s['w']]; TLC = [s for s in TLC if len(s['w']) >= 10]
# overlap: same lyricist and at least half of the LD song's distinct words found in the TLC song
idx = collections.defaultdict(list)
for j, t in enumerate(TLC): idx[t['lyr']].append(j)
over_t = set(); over_l = set()
for i, s in enumerate(LD):
    a = set(s['w'])
    for j in idx.get(s['lyr'], []):
        b = set(TLC[j]['w'])
        if len(a & b) / max(1, min(len(a), len(b))) >= 0.5:
            over_t.add(j); over_l.add(i)
TLCx = [t for j, t in enumerate(TLC) if j not in over_t]
def film(ss): return [s for s in ss if s['y'] and s['y'] >= 1950 and 'Bharathi' not in s['lyr'] and 'Siddhar' not in s['lyr']]
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]
R = np.random.default_rng(7)
def nee(ss):
    k = sum(any(is_you(x) for x in s['w']) for s in ss); return {'v': round(100 * k / len(ss), 1), 'n': len(ss), 'ci': wilson(k, len(ss))}
def wlen(ss):
    L = [np.array([len(w) for w in s['w']]) for s in ss]
    bs = [np.concatenate([L[i] for i in R.integers(0, len(L), len(L))]).mean() for _ in range(1000)]
    return {'v': round(float(np.concatenate(L).mean()), 2), 'n': len(ss), 'ci': [round(float(np.percentile(bs, 2.5)), 2), round(float(np.percentile(bs, 97.5)), 2)]}
def eras(ss, f): return [f([s for s in ss if a <= s['y'] <= b]) for a, b in ERAS]
def career(ss, poet, f):
    p = sorted([s for s in ss if s['lyr'] == poet], key=lambda s: s['y']); h = len(p) // 2
    return {'poet': poet, 'n': len(p), 'from': [p[0]['y'], p[h - 1]['y']], 'to': [p[h]['y'], p[-1]['y']], 'a': f(p[:h]), 'b': f(p[h:])}
out = {'tlc_songs': len(TLC), 'tlc_dated_film': len(film(TLC)), 'overlap_tlc': len(over_t), 'overlap_ld': len(over_l), 'trends': {}}
POETS = ['Kannadasan', 'Vaali', 'Vairamuthu', 'Na. Muthukumar']
for name, ss in [('LD', film(LD)), ('TLC', film(TLC)), ('TLC_minus_overlap', film(TLCx))]:
    out['trends'][name] = {'nee': eras(ss, nee), 'len': eras(ss, wlen),
                           'careers_nee': [career(ss, p, nee) for p in POETS if sum(s['lyr'] == p for s in ss) >= 20],
                           'careers_len': [career(ss, p, wlen) for p in POETS if sum(s['lyr'] == p for s in ss) >= 20]}
    dec = collections.defaultdict(list)
    for s in ss: dec[s['y'] // 10 * 10].append(s)
    out['trends'][name]['decades'] = {d: {'n': len(g), 'nee': nee(g)['v'], 'len': wlen(g)['v']} for d, g in sorted(dec.items()) if len(g) >= 15}
# lyricist identification
def model(): return make_pipeline(TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 3), min_df=2, sublinear_tf=True), LinearSVC(C=0.3, class_weight='balanced'))
SIX = ['Kannadasan', 'Vaali', 'Vairamuthu', 'Na. Muthukumar', 'Thamarai', 'Madhan Karky']
def cv(ss, cls, seed=7):
    ss = [s for s in ss if s['lyr'] in cls]; y = np.array([s['lyr'] for s in ss])
    p = cross_val_predict(model(), [s['doc'] for s in ss], y, cv=StratifiedKFold(10, shuffle=True, random_state=seed))
    rc = {c: round(float(np.mean(p[y == c] == c)), 3) for c in cls}
    return {'bacc': round(float(balanced_accuracy_score(y, p)), 3), 'n': len(ss), 'counts': {c: int((y == c).sum()) for c in cls}, 'recall': rc}
ld_counts = collections.Counter(s['lyr'] for s in LD if s['lyr'] in SIX)
sub = []
for seed in range(10):
    rr = np.random.default_rng(seed); pick = []
    for c in SIX:
        pool = [s for s in TLCx if s['lyr'] == c]; pick += [pool[i] for i in rr.choice(len(pool), ld_counts[c], replace=False)]
    sub.append(cv(pick, SIX, seed)['bacc'])
m = model(); tr = [s for s in TLCx if s['lyr'] in SIX]; m.fit([s['doc'] for s in tr], [s['lyr'] for s in tr])
te = [s for i, s in enumerate(LD) if s['lyr'] in SIX]
pr = m.predict([s['doc'] for s in te]); yt = np.array([s['lyr'] for s in te])
out['lyricist_id'] = {'LD_6': cv(LD, SIX), 'TLC_6': cv(TLCx, SIX),
                      'TLC_6_subsampled_to_LD_sizes': {'bacc_mean': round(float(np.mean(sub)), 3), 'bacc_min': round(min(sub), 3), 'bacc_max': round(max(sub), 3), 'runs': 10},
                      'train_TLC_test_LD_6': {'bacc': round(float(balanced_accuracy_score(yt, pr)), 3), 'n_test': len(te),
                                              'recall': {c: round(float(np.mean(pr[yt == c] == c)), 3) for c in SIX}},
                      'TLC_binary_Vaali_Kannadasan': cv(TLCx, ['Vaali', 'Kannadasan']),
                      'LD_binary_Vaali_Kannadasan': cv(LD, ['Vaali', 'Kannadasan']),
                      'chance_6': round(1 / 6, 3)}
# length control for the pronoun trend: rate per 1000 tokens, and share among songs of 50-120 tokens
out['nee_length_control'] = {}
for name, ss in [('LD', film(LD)), ('TLC_minus_overlap', film(TLCx))]:
    rows = []
    for a, b in ERAS:
        g = [s for s in ss if a <= s['y'] <= b]; m = [s for s in g if 50 <= len(s['w']) <= 120]
        rows.append({'per_1000_tokens': round(1000 * sum(sum(is_you(x) for x in s['w']) for s in g) / sum(len(s['w']) for s in g), 1),
                     'mean_tokens': round(float(np.mean([len(s['w']) for s in g]))),
                     'share_len_50_120': nee(m) if m else None})
    out['nee_length_control'][name] = rows
if __name__ == "__main__": json.dump(out, open(os.path.join(OUT, "robustness.json"), 'w'), ensure_ascii=False, indent=1)
if __name__ == "__main__": print(json.dumps(out, ensure_ascii=False, indent=1))
