"""Meaning-space checks on TLC with bge-m3 (Tamil only; TLC has no translations).
Samples up to 120 songs per lyricist for the 14 film lyricists in the LD-690 family tree
(overlap with LD-690 removed), embeds them, and compares with LD-690's Tamil-only vectors:
 (1) AMI of k-means clusters with poet and 20-year era (TLC has no mood labels);
 (2) each poet's nearest poet (cosine of mean vectors) and its bootstrap stability.
Writes out/tlc_meaning.json; caches vectors in paper/ext/tlc_m3.npy."""
import json, os, sys, collections, time
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score as ami
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'paper', 'analysis')); sys.path.insert(0, os.path.join(ROOT, 'scripts', 'songmap'))
import robustness as rb
import embed_m3 as em
TREE = ['Vairamuthu', 'Kannadasan', 'Vaali', 'Na. Muthukumar', 'Thamarai', 'Madhan Karky', 'Gangai Amaran', 'Vivek',
        'Pulamaipithan', 'Kabilan', 'Pa.Vijay', 'Yugabharathi', 'Kamal Hassan', 'Pazhani Bharathi']
NAME = {'Kamal Hassan': 'Kamal haasan', 'Pazhani Bharathi': 'Pazhani bharati'}   # LD spellings
R = np.random.default_rng(7)
T = [t for j, t in enumerate(rb.TLC) if j not in rb.over_t]
pick = []
for p in TREE:
    pool = [t for t in T if t['lyr'] == p]
    pick += [pool[i] for i in sorted(R.choice(len(pool), min(120, len(pool)), replace=False))]
cache = os.path.join(ROOT, 'paper', 'ext', 'tlc_m3.npy')
if os.path.exists(cache) and len(np.load(cache)) == len(pick):
    V = np.load(cache)
else:
    tok, sess = em.load(); V = []; t0 = time.time()
    for n, t in enumerate(pick):
        cs = em.chunks(em.uniq(t['doc'].split('\n')), tok)
        v = np.mean([em.embed(tok, sess, c) for c in cs], 0); V.append(v / np.linalg.norm(v))
        if n % 100 == 0: print(n, len(pick), round(time.time() - t0), 's', flush=True)
    V = np.array(V, dtype=np.float32); np.save(cache, V)
def centre(X): X = X - X.mean(0); return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
out = {'n': len(pick), 'per_poet': dict(collections.Counter(t['lyr'] for t in pick))}
# (1) clusters vs poet and era, TLC (7 big poets, dated) and LD (same poets, Tamil-only vectors)
SEVEN = ['Vairamuthu', 'Kannadasan', 'Vaali', 'Na. Muthukumar', 'Thamarai', 'Madhan Karky', 'Pa.Vijay']
def amis(X, poet, era, k=7):
    r = np.mean([[ami(poet, KMeans(k, n_init=5, random_state=s).fit_predict(X)), ami(era, KMeans(k, n_init=5, random_state=s).fit_predict(X))] for s in range(5)], 0)
    return {'poet': round(float(r[0]), 3), 'era': round(float(r[1]), 3)}
ix = [i for i, t in enumerate(pick) if t['lyr'] in SEVEN and t['y']]
X = centre(V)[ix]
out['ami_tlc'] = amis(X, [pick[i]['lyr'] for i in ix], [max(1940, pick[i]['y'] // 20 * 20) for i in ix]); out['ami_tlc']['n'] = len(ix)
SM = os.path.join(ROOT, 'data', 'songmap')
ids = json.load(open(os.path.join(SM, 'vector_ids.json'))); S = {str(s['id']): s for s in json.load(open(os.path.join(SM, 'songs.json')))}
LT = centre(np.load(os.path.join(SM, 'vectors_m3_ta.npy')))
lrow = []
for r, i in enumerate(ids):
    s = S[str(i)]; l = rb.norm(s['lyricist'])
    try: y = int(str(s['year'])[:4])
    except Exception: y = None
    lrow.append((r, l, y))
jx = [r for r, l, y in lrow if l in SEVEN and y]
out['ami_ld_tamil_only'] = amis(LT[jx], [lrow[r][1] for r in jx], [max(1940, lrow[r][2] // 20 * 20) for r in jx]); out['ami_ld_tamil_only']['n'] = len(jx)
# (2) nearest poet and stability, TLC vs LD (Tamil-only)
def nearest(Xc, groups, names, boot=0):
    def nn(rows):
        C = np.array([Xc[rows[n]].mean(0) for n in names]); C /= np.linalg.norm(C, axis=1, keepdims=True)
        sim = C @ C.T; np.fill_diagonal(sim, -9); return {n: names[int(sim[i].argmax())] for i, n in enumerate(names)}
    base = nn(groups); keep = collections.Counter()
    for _ in range(boot):
        b = nn({n: list(R.choice(groups[n], len(groups[n]))) for n in names})
        for n in names: keep[n] += b[n] == base[n]
    return {n: {'nearest': base[n], 'stable': round(keep[n] / boot, 2) if boot else None} for n in names}
Xc = centre(V); g = collections.defaultdict(list)
for i, t in enumerate(pick): g[t['lyr']].append(i)
names = [p for p in TREE if len(g[p]) >= 9]
out['tlc_nearest'] = nearest(Xc, g, names, 500)
lg = collections.defaultdict(list)
for r, l, y in lrow: lg[l].append(r)
lnames = [n for n in [NAME.get(p, p) for p in TREE] if len(lg[n]) >= 9]
out['ld_tamil_nearest'] = nearest(LT, lg, lnames, 500)
json.dump(out, open(os.path.join(rb.OUT, 'tlc_meaning.json'), 'w'), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in out.items()}, ensure_ascii=False, indent=0))
