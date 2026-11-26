"""2D layout + themes for the song map.

UMAP (cosine) places songs so that translations with similar meaning sit
close together. KMeans on an 8-D UMAP of the vectors finds the themes; each theme's
distinctive words come from class-based TF-IDF. Names for the themes live in
data/songmap/themes.json (hand-written, keyed by theme number) so a re-run
keeps them as long as the seed and data are unchanged.
"""
import json, os, re, sys
from collections import Counter
import numpy as np
import umap
from sklearn.cluster import KMeans

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SM = os.path.join(ROOT, 'data', 'songmap')
K = int(os.environ.get('K', 10))

STOP = set('''a about above after again against all am an and any are as at be because been before being below between both but by
can could did do does doing down during each few for from further had has have having he her here hers herself him himself his how
i if in into is it its itself just me more most my myself no nor not now of off on once only or other our ours ourselves out over own
same she should so some such than that the their theirs them themselves then there these they this those through to too under until
up very was we were what when where which while who whom why will with would you your yours yourself yourselves oh o let like one ever
even every shall may might us thy thee thou also yet upon within without dont don't im i'm it's that's let's can't won't  ll ve re s t
come go get got make made see say tell know way take give thing things much many will']'''.split())

def tokens(t):
    return [w for w in re.findall(r"[a-z][a-z']+", t.lower()) if w not in STOP and len(w) > 2]

def main():
    V = np.load(os.path.join(SM, 'vectors.npy'))
    ids = json.load(open(os.path.join(SM, 'vector_ids.json')))
    S = {s['id']: s for s in json.load(open(os.path.join(SM, 'songs.json')))}
    xy = umap.UMAP(n_neighbors=15, min_dist=0.3, spread=1.2, metric='cosine', random_state=7).fit_transform(V)
    xy = (xy - xy.min(0)) / (xy.max(0) - xy.min(0))
    # cluster in an 8-D UMAP space (keeps neighbourhoods, so themes stay together on the 2-D map)
    # themes are found without labels in the bge-m3 meaning space when it exists (Tamil + English),
    # else in the MiniLM English space
    mp = os.path.join(SM, 'vectors_m3.npy')
    C = V
    if os.path.exists(mp):
        C = np.load(mp); C = C - C.mean(0); C /= np.linalg.norm(C, axis=1, keepdims=True)
    z = umap.UMAP(n_neighbors=15, n_components=8, min_dist=0.0, metric='cosine', random_state=7).fit_transform(C)
    lab = KMeans(n_clusters=K, n_init=20, random_state=7).fit_predict(z)
    # class-based tf-idf over English words
    docs = [Counter() for _ in range(K)]
    df = Counter()
    for i, pid in enumerate(ids):
        toks = tokens(' '.join(S[pid]['english']))
        docs[lab[i]].update(toks)
        df.update(set(toks))
    N = len(ids)
    terms = {}
    for k in range(K):
        tot = sum(docs[k].values()) or 1
        sc = {w: (c / tot) * np.log(1 + N / df[w]) for w, c in docs[k].items() if df[w] >= 3}
        terms[k] = [w for w, _ in sorted(sc.items(), key=lambda x: -x[1])[:14]]
    sims = V @ V.T
    np.fill_diagonal(sims, -1)
    nn = np.argsort(-sims, axis=1)[:, :6]
    cent = np.vstack([C[lab == k].mean(0) for k in range(K)])
    cent /= np.linalg.norm(cent, axis=1, keepdims=True)
    out = {'points': [], 'clusters': []}
    for i, pid in enumerate(ids):
        out['points'].append({'id': pid, 'x': round(float(xy[i, 0]), 4), 'y': round(float(xy[i, 1]), 4),
                              'c': int(lab[i]), 'nn': [int(ids[j]) for j in nn[i]],
                              'nns': [round(float(sims[i, j]), 3) for j in nn[i]]})
    for k in range(K):
        mem = np.where(lab == k)[0]
        core = mem[np.argsort(-(C[mem] @ cent[k]))][:8]
        out['clusters'].append({'k': k, 'n': int(len(mem)), 'terms': terms[k],
                                'core': [S[ids[j]]['title'] for j in core]})
    # Tamil wording view: same songs, laid out from the Tamil style vectors
    tp = os.path.join(SM, 'vectors_ta.npy')
    if os.path.exists(tp):
        Z = np.load(tp)
        txy = umap.UMAP(n_neighbors=15, min_dist=0.3, spread=1.2, metric='cosine', random_state=7).fit_transform(Z)
        txy = (txy - txy.min(0)) / (txy.max(0) - txy.min(0))
        ts = Z @ Z.T
        np.fill_diagonal(ts, -1)
        tnn = np.argsort(-ts, axis=1)[:, :6]
        for i, pt in enumerate(out['points']):
            pt['tx'] = round(float(txy[i, 0]), 4); pt['ty'] = round(float(txy[i, 1]), 4)
            pt['tnn'] = [int(ids[j]) for j in tnn[i]]
            pt['tnns'] = [round(float(ts[i, j]), 3) for j in tnn[i]]
    # Meaning view (bge-m3, Tamil + English, centred): layout and nearest songs
    mp = os.path.join(SM, 'vectors_m3.npy')
    if os.path.exists(mp):
        M = np.load(mp); M = M - M.mean(0); M /= np.linalg.norm(M, axis=1, keepdims=True)
        mxy = umap.UMAP(n_neighbors=15, min_dist=0.25, spread=1.2, metric='cosine', random_state=7).fit_transform(M)
        mxy = (mxy - mxy.min(0)) / (mxy.max(0) - mxy.min(0))
        ms = M @ M.T; np.fill_diagonal(ms, -1)
        mnn = np.argsort(-ms, axis=1)[:, :6]
        for i, pt in enumerate(out['points']):
            pt['mx'] = round(float(mxy[i, 0]), 4); pt['my'] = round(float(mxy[i, 1]), 4)
            pt['nn'] = [int(ids[j]) for j in mnn[i]]
            pt['nns'] = [round(float(ms[i, j]), 3) for j in mnn[i]]
    json.dump(out, open(os.path.join(SM, 'layout.json'), 'w'), ensure_ascii=False)
    for c in out['clusters']:
        print(c['k'], c['n'], ', '.join(c['terms']))
        print('    ', ' | '.join(c['core']))

if __name__ == '__main__':
    main()
