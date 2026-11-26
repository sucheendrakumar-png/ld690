"""Extra checks for the paper (no network):
 (a) 7-way lyricist ID on LD-690: 95% bootstrap CI of balanced accuracy, spread over 10 CV seeds,
     and a year-only baseline (how far could era alone go?);
 (b) Vaali vs Kannadasan restricted to the years both were active (<=1981), LD-690 and TLC;
 (c) stability of the poets' family tree: for each poet, how often its nearest poet stays the same
     over 500 bootstrap resamples of songs.
Writes paper/analysis/out/extra_checks.json."""
import json, os, sys, csv, collections
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'paper', 'analysis'))
import robustness as rb   # reuses its corpora and model definition (it re-runs, ~1 min)
SM = os.path.join(ROOT, 'data', 'songmap')
out = {}
SEVEN = ['Kannadasan', 'Vaali', 'Vairamuthu', 'Na. Muthukumar', 'Thamarai', 'Madhan Karky', 'Subramanya Bharathi']
ss = [s for s in rb.LD if s['lyr'] in SEVEN]; y = np.array([s['lyr'] for s in ss]); docs = [s['doc'] for s in ss]
accs, preds = [], None
for seed in range(10):
    p = cross_val_predict(rb.model(), docs, y, cv=StratifiedKFold(10, shuffle=True, random_state=seed))
    accs.append(balanced_accuracy_score(y, p))
    if seed == 7: preds = p
R = np.random.default_rng(1); bs = []
for _ in range(2000):
    i = R.integers(0, len(y), len(y)); bs.append(balanced_accuracy_score(y[i], preds[i]))
out['ld7'] = {'n': len(y), 'bacc_seed7': round(float(balanced_accuracy_score(y, preds)), 3),
              'ci95': [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)],
              'seeds_mean': round(float(np.mean(accs)), 3), 'seeds_min': round(float(min(accs)), 3), 'seeds_max': round(float(max(accs)), 3)}
# year-only baseline on dated songs of the seven (Bharathi's poems are mostly undated -> excluded)
d = [s for s in ss if s['y']]; yd = np.array([s['lyr'] for s in d]); X = np.array([[s['y']] for s in d])
p = cross_val_predict(DecisionTreeClassifier(max_depth=4, class_weight='balanced', random_state=0), X, yd, cv=StratifiedKFold(10, shuffle=True, random_state=7))
pt = cross_val_predict(rb.model(), [s['doc'] for s in d], yd, cv=StratifiedKFold(10, shuffle=True, random_state=7))
out['year_only'] = {'n': len(d), 'classes': sorted(set(yd.tolist())), 'bacc_year_only': round(float(balanced_accuracy_score(yd, p)), 3),
                    'bacc_text_same_songs': round(float(balanced_accuracy_score(yd, pt)), 3)}
# era-matched binary
for name, corpus in [('LD', rb.LD), ('TLC', rb.TLCx)]:
    e = [s for s in corpus if s['lyr'] in ('Vaali', 'Kannadasan') and s['y'] and 1954 <= s['y'] <= 1981]
    out[f'binary_era_matched_{name}'] = rb.cv(e, ['Vaali', 'Kannadasan'])
# family-tree stability (bge-m3 meaning space, poets with 9+ songs, as on the Song Map)
ids = json.load(open(os.path.join(SM, 'vector_ids.json')))
S = {str(s['id']): s for s in json.load(open(os.path.join(SM, 'songs.json')))}
M = np.load(os.path.join(SM, 'vectors_m3.npy')); M = M - M.mean(0); M /= np.linalg.norm(M, axis=1, keepdims=True)
by = collections.defaultdict(list)
for r, i in enumerate(ids): by[rb.norm(S[str(i)]['lyricist'])].append(r)
poets = sorted([n for n, v in by.items() if len(v) >= 9], key=lambda n: -len(by[n]))
def nearest(rows):
    C = np.array([M[rows[n]].mean(0) for n in poets]); C /= np.linalg.norm(C, axis=1, keepdims=True)
    sim = C @ C.T; np.fill_diagonal(sim, -9); return {n: poets[int(sim[i].argmax())] for i, n in enumerate(poets)}
base = nearest({n: by[n] for n in poets}); keep = collections.Counter()
for _ in range(500):
    nb = nearest({n: list(R.choice(by[n], len(by[n]))) for n in poets})
    for n in poets: keep[n] += nb[n] == base[n]
out['tree'] = [{'poet': n, 'n': len(by[n]), 'nearest': base[n], 'stable': round(keep[n] / 500, 2)} for n in poets]
json.dump(out, open(os.path.join(rb.OUT, 'extra_checks.json'), 'w'), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
