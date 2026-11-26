"""Do the songs cluster by poet without being told? KMeans on each reading, then adjusted
mutual information of the clusters with poet, mood and 20-year era (0 = none, 1 = perfect).
Songs by the seven poets with 25+ songs that have a mood and a year. Writes cluster_check.json."""
import json, os, ast
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score as ami
SM = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'songmap')
ids = json.load(open(os.path.join(SM, 'vector_ids.json')))
S = {int(s['id']): s for s in json.load(open(os.path.join(SM, 'songs.json')))}
LYR = {'Vasana Kavidhai': 'Subramanya Bharathi', 'Na.Muthukumar': 'Na. Muthukumar'}
TOP = {'Vairamuthu', 'Kannadasan', 'Subramanya Bharathi', 'Vaali', 'Na. Muthukumar', 'Thamarai', 'Madhan Karky'}
rows = []
for i, x in enumerate(ids):
    s = S[int(x)]; l = LYR.get(s['lyricist'], s['lyricist'])
    m = ast.literal_eval(s['moods']) if isinstance(s['moods'], str) else s['moods']
    try: y = int(str(s['year'])[:4])
    except Exception: y = None
    if l in TOP and m and y: rows.append((i, l, m[0], max(1940, y // 20 * 20)))
idx = [r[0] for r in rows]
labs = {'poet': [r[1] for r in rows], 'mood': [r[2] for r in rows], 'era': [r[3] for r in rows]}
out = {'n': len(rows), 'k': 7, 'readings': {}}
for key, name in [('meaning', 'vectors_m3.npy'), ('wording', 'vectors_ta.npy'), ('english', 'vectors.npy')]:
    X = np.load(os.path.join(SM, name))[idx]; X = X - X.mean(0); X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    r = np.mean([[ami(labs[n], KMeans(7, n_init=5, random_state=sd).fit_predict(X)) for n in labs] for sd in range(5)], 0)
    out['readings'][key] = {n: round(float(v), 2) for n, v in zip(labs, r)}
json.dump(out, open(os.path.join(SM, 'cluster_check.json'), 'w'), indent=1)
print(out)
