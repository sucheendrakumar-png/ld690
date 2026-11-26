"""What the English translations carry. Writes paper/analysis/out/translation.json.

(1) Poet fingerprints in the English: the paper's lyricist model (TF-IDF char_wb 1-3 + LinearSVC,
    10-fold, balanced accuracy) trained on the English translations alone, on the Tamil alone
    (same songs, same folds), and on the mood labels alone (a topic-only baseline).
(2) Meaning by language: k-means (k=7) on bge-m3 vectors of the Tamil only, the English only,
    and their average; adjusted mutual information with poet, mood and era, exactly as
    scripts/songmap/cluster_check.py.
(3) The translations' profile over the whole corpus: line and token ratios, English end rhyme
    (CMU dictionary), and register by mood (archaic/elevated diction, capitalised divine pronouns).
(4) Against machine translation, on the line-aligned sample in paper/ext/mt_pairs.json
    (paper/analysis/mt_baseline.py): length, vocabulary, rhyme, register, and bge-m3 similarity
    of each English line to its Tamil line.
"""
import ast, json, os, re, sys, collections
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score as ami
import pronouncing

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SM = os.path.join(ROOT, 'data', 'songmap'); EXT = os.path.join(ROOT, 'paper', 'ext')
OUT = os.path.join(ROOT, 'paper', 'analysis', 'out')
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'songmap'))
from tamil_style import tamil_doc
MERGE = {'Vasana Kavidhai': 'Subramanya Bharathi'}
rng = np.random.default_rng(7)

def L(t): return [l.strip() for l in t.split('\n') if l.strip()]
def uniq(ls):
    seen, out = set(), []
    for l in ls:
        k = l.lower()
        if k not in seen: seen.add(k); out.append(l)
    return out
def moods(s): return ast.literal_eval(s['moods']) if isinstance(s['moods'], str) else (s['moods'] or [])

S = {s['id']: s for s in json.load(open(os.path.join(SM, 'songs.json')))}
ids = json.load(open(os.path.join(SM, 'vector_ids.json')))
par = [i for i in ids if S[i]['english'] and S[i]['tamil'].strip()]
lab = {i: MERGE.get(S[i]['lyricist'], S[i]['lyricist']) for i in par}
cnt = collections.Counter(lab.values()); big = sorted(l for l, n in cnt.items() if n >= 25)
out = {}

# ---------- (1) poet fingerprints in the English ----------
def model(analyzer='char_wb', ngr=(1, 3)):
    return make_pipeline(TfidfVectorizer(analyzer=analyzer, ngram_range=ngr, min_df=2, sublinear_tf=True, lowercase=True),
                         LinearSVC(C=0.3, class_weight='balanced'))
I = [i for i in par if lab[i] in big]; y = np.array([lab[i] for i in I])
en_doc = ['\n'.join(uniq([l for b in S[i]['english'] for l in L(b)])) for i in I]
ta_doc = [tamil_doc(S[i]['tamil']) for i in I]
mlb = MultiLabelBinarizer(); Xm = mlb.fit_transform([moods(S[i]) for i in I])
def run(docs, m, seed=7):
    cv = StratifiedKFold(10, shuffle=True, random_state=seed)
    p = cross_val_predict(m, docs, y, cv=cv)
    return p, balanced_accuracy_score(y, p)
res = {}
for name, docs, mk in [('tamil_char', ta_doc, lambda: model()), ('english_char', en_doc, lambda: model()),
                       ('english_words', en_doc, lambda: model('word', (1, 2))),
                       ('both', [a + '\n' + b for a, b in zip(ta_doc, en_doc)], lambda: model())]:
    accs = [run(docs, mk(), sd)[1] for sd in range(10)]
    p, a = run(docs, mk())
    rec = {c: round(float(np.mean(p[y == c] == c)), 3) for c in big}
    res[name] = {'bacc': round(a, 3), 'seeds': [round(min(accs), 3), round(max(accs), 3)], 'recall': rec}
    if name == 'english_char':
        res[name]['confusion'] = confusion_matrix(y, p, labels=big, normalize='true').round(3).tolist()
pm = cross_val_predict(LinearSVC(C=0.3, class_weight='balanced'), Xm, y, cv=StratifiedKFold(10, shuffle=True, random_state=7))
res['moods_only'] = {'bacc': round(balanced_accuracy_score(y, pm), 3)}
# agreement: does the English model err where the Tamil model errs?
pt, _ = run(ta_doc, model()); pe, _ = run(en_doc, model())
res['agree'] = {'both_right': round(float(np.mean((pt == y) & (pe == y))), 3), 'same_prediction': round(float(np.mean(pt == pe)), 3)}
# Vaali vs Kannadasan
vk = np.isin(y, ['Vaali', 'Kannadasan'])
for name, docs in [('tamil', ta_doc), ('english', en_doc)]:
    d = [x for x, k in zip(docs, vk) if k]; yy = y[vk]
    p = cross_val_predict(model(), d, yy, cv=StratifiedKFold(10, shuffle=True, random_state=7))
    res[f'vaali_kannadasan_{name}'] = round(balanced_accuracy_score(yy, p), 3)
# controls for topic and song choice: without Bharathi's prose-poem series, and within one mood
def ev(sub, which, k=5):
    yy = np.array([lab[i] for i in sub]); c = collections.Counter(yy)
    keep = [j for j, l in enumerate(yy) if c[l] >= k]; sub = [sub[j] for j in keep]; yy = yy[keep]
    docs = ['\n'.join(uniq([l for b in S[i]['english'] for l in L(b)])) if which == 'en' else tamil_doc(S[i]['tamil']) for i in sub]
    accs = [balanced_accuracy_score(yy, cross_val_predict(model(), docs, yy, cv=StratifiedKFold(k, shuffle=True, random_state=sd))) for sd in range(5)]
    return {'n': len(sub), 'k': len(set(yy)), 'bacc': round(float(np.mean(accs)), 3), 'chance': round(1 / len(set(yy)), 3)}
m0 = lambda i: moods(S[i])[0] if moods(S[i]) else None
res['controls'] = {}
for name, sub in [('no_bharathi', [i for i in I if lab[i] != 'Subramanya Bharathi']),
                  ('love_only', [i for i in I if m0(i) == 'love']),
                  ('philosophy_only', [i for i in I if m0(i) == 'philosophy']),
                  ('love_no_bharathi', [i for i in I if m0(i) == 'love' and lab[i] != 'Subramanya Bharathi'])]:
    res['controls'][name] = {'english': ev(sub, 'en'), 'tamil': ev(sub, 'ta')}
# the translator's voice: who posted each translation (WordPress author; data/post_authors.json,
# fetched from /wp/v2/posts?_fields=id,author), the three team members with 100+ songs
A = {int(k): v for k, v in json.load(open(os.path.join(ROOT, 'data', 'post_authors.json'))).items()}
tr = {i: A.get(int(i)) for i in par}
top = [a for a, n in collections.Counter(tr.values()).most_common(3)]
T3 = [i for i in par if tr[i] in top]
def ev_lab(sub, which, labf, k=5):
    yy = np.array([str(labf(i)) for i in sub]); c = collections.Counter(yy)
    keep = [j for j, l in enumerate(yy) if c[l] >= k]; sub = [sub[j] for j in keep]; yy = yy[keep]
    docs = ['\n'.join(uniq([l for b in S[i]['english'] for l in L(b)])) if which == 'en' else tamil_doc(S[i]['tamil']) for i in sub]
    accs = [balanced_accuracy_score(yy, cross_val_predict(model(), docs, yy, cv=StratifiedKFold(k, shuffle=True, random_state=sd))) for sd in range(5)]
    return {'n': len(sub), 'k': len(set(yy)), 'bacc': round(float(np.mean(accs)), 3), 'chance': round(1 / len(set(yy)), 3)}
res['translator'] = {'songs_per_translator': [collections.Counter(tr.values())[a] for a in top],
                     'from_english': ev_lab(T3, 'en', lambda i: tr[i]), 'from_tamil': ev_lab(T3, 'ta', lambda i: tr[i]),
                     'poet_within_translator': [{'english': ev_lab([i for i in I if tr[i] == a], 'en', lambda i: lab[i]),
                                                 'tamil': ev_lab([i for i in I if tr[i] == a], 'ta', lambda i: lab[i])} for a in top[:2]]}
res['n'] = len(I); res['classes'] = big
out['id'] = res
print('ID', json.dumps(res, indent=0)[:1500])

# ---------- (2) meaning by language ----------
TOP = set(big)
rows = []
for r, i in enumerate(ids):
    s = S[i]; l = MERGE.get(s['lyricist'], s['lyricist']); m = moods(s)
    try: yr = int(str(s['year'])[:4])
    except Exception: yr = None
    if l in TOP and m and yr: rows.append((r, l, m[0], max(1940, yr // 20 * 20)))
idx = [r[0] for r in rows]
labs = {'poet': [r[1] for r in rows], 'mood': [r[2] for r in rows], 'era': [r[3] for r in rows]}
mean = {}
for key, f in [('tamil', 'vectors_m3_ta.npy'), ('english', 'vectors_m3_en.npy'), ('both', 'vectors_m3.npy')]:
    X = np.load(os.path.join(SM, f))[idx]; X = X - X.mean(0); X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    r = np.mean([[ami(labs[n], KMeans(7, n_init=5, random_state=sd).fit_predict(X)) for n in labs] for sd in range(5)], 0)
    mean[key] = {n: round(float(v), 3) for n, v in zip(labs, r)}
# how close is each song's Tamil to its own English, against other songs' English?
Ta = np.load(os.path.join(SM, 'vectors_m3_ta.npy')); En = np.load(os.path.join(SM, 'vectors_m3_en.npy'))
Ta = Ta / np.linalg.norm(Ta, axis=1, keepdims=True); En = En / np.linalg.norm(En, axis=1, keepdims=True)
keep = [r for r, i in enumerate(ids) if i in set(par)]
Ss = Ta[keep] @ En[keep].T
rank1 = float(np.mean(Ss.argmax(1) == np.arange(len(keep))))
mean['retrieval'] = {'n': len(keep), 'top1': round(rank1, 3),
                     'own': round(float(np.diag(Ss).mean()), 3), 'other': round(float(Ss[~np.eye(len(keep), dtype=bool)].mean()), 3)}
mean['n'] = len(rows)
out['meaning'] = mean
print('MEANING', mean)

# ---------- (3) profile of the translations ----------
ARCH = {'thou', 'thee', 'thy', 'thine', 'ye', 'hath', 'doth', 'art', 'shalt', 'wilt', "o'er", 'ere', 'whence', 'thence',
        'hither', 'thither', 'yonder', 'nay', 'lo', 'behold', 'oft', 'naught', 'aught', 'unto', 'amidst', 'amongst', 'beseech'}
DIV = re.compile(r"(?<=\w[ ,;:])(He|Him|His|Thou|Thee|Thy|Thine|You|Your|Yours|Lord)\b")
WORD = re.compile(r"[A-Za-z']+")
def last_word(l):
    w = WORD.findall(l)
    return w[-1].lower().strip("'") if w else ''
def rhymes(a, b):
    if not a or not b or a == b: return False
    pa, pb = pronouncing.phones_for_word(a), pronouncing.phones_for_word(b)
    if not pa or not pb: return False
    return any(pronouncing.rhyming_part(x) == pronouncing.rhyming_part(y_) for x in pa for y_ in pb)
def profile(lines_by_song):
    toks = [w.lower() for ls in lines_by_song for l in ls for w in WORD.findall(l)]
    pairs = [(a, b) for ls in lines_by_song for a, b in zip(ls, ls[1:])]
    known = [(last_word(a), last_word(b)) for a, b in pairs]
    known = [(a, b) for a, b in known if pronouncing.phones_for_word(a) and pronouncing.phones_for_word(b) and a != b]
    nl = sum(len(ls) for ls in lines_by_song)
    return {'lines': nl, 'tokens': len(toks), 'tokens_per_line': round(len(toks) / max(1, nl), 2),
            'arch_per_1k': round(1000 * sum(t in ARCH for t in toks) / max(1, len(toks)), 2),
            'divine_caps_per_1k_lines': round(1000 * sum(len(DIV.findall(l)) for ls in lines_by_song for l in ls) / max(1, nl), 1),
            'end_rhyme': round(float(np.mean([rhymes(a, b) for a, b in known])) if known else 0, 3),
            'question_lines': round(sum(l.rstrip().endswith('?') for ls in lines_by_song for l in ls) / max(1, nl), 3),
            'ttr_2k': round(float(np.mean([len(set(rng.choice(toks, 2000, replace=False))) / 2000 for _ in range(50)])), 3) if len(toks) >= 2000 else None}
corp = {}
en_lines = {i: uniq([l for b in S[i]['english'] for l in L(b)]) for i in par}
ta_lines = {i: uniq(L(S[i]['tamil'])) for i in par}
corp['all'] = profile([en_lines[i] for i in par])
corp['ta_lines'] = sum(len(v) for v in ta_lines.values()); corp['en_lines'] = sum(len(v) for v in en_lines.values())
corp['ta_tokens'] = sum(len(re.findall(r'[஀-௿]+', ' '.join(v))) for v in ta_lines.values())
for m in ['devotion', 'philosophy', 'love', 'longing', 'joy', 'nature', 'mother', 'courage', 'homeland', 'friendship']:
    ss = [i for i in par if moods(S[i]) and moods(S[i])[0] == m]
    if len(ss) >= 15: corp[m] = dict(profile([en_lines[i] for i in ss]), n=len(ss))
out['profile'] = corp
print('PROFILE', json.dumps(corp)[:1200])

# ---------- (4) against machine translation ----------
mp = os.path.join(EXT, 'mt_pairs.json')
if os.path.exists(mp):
    P = json.load(open(mp))
    ld = [[x['ld'] for x in s['lines']] for s in P]; mt = [[x['mt'] for x in s['lines']] for s in P]
    cmp = {'songs': len(P), 'lines': sum(len(s['lines']) for s in P), 'ld': profile(ld), 'mt': profile(mt)}
    for m in ['devotion', 'love']:
        ss = [k for k, s in enumerate(P) if moods(s) and moods(s)[0] == m]
        if ss: cmp[m] = {'n': len(ss), 'ld': profile([ld[k] for k in ss]), 'mt': profile([mt[k] for k in ss])}
    sys.path.insert(0, os.path.join(ROOT, 'scripts', 'songmap'))
    import embed_m3 as em
    tok, sess = em.load()
    def E(t):
        v = np.asarray(em.embed(tok, sess, t), dtype=np.float32); return v / np.linalg.norm(v)
    cache_p = os.path.join(EXT, 'mt_m3.npz')
    flat = [x for s in P for x in s['lines']]
    if os.path.exists(cache_p) and np.load(cache_p)['ta'].shape[0] == len(flat):
        z = np.load(cache_p); vt, vl, vm = z['ta'], z['ld'], z['mt']
    else:
        vt = np.array([E(x['ta']) for x in flat]); vl = np.array([E(x['ld']) for x in flat]); vm = np.array([E(x['mt']) for x in flat])
        np.savez(cache_p, ta=vt, ld=vl, mt=vm)
    s_ld = (vt * vl).sum(1); s_mt = (vt * vm).sum(1); s_lm = (vl * vm).sum(1)
    # line retrieval within song: is each English line closest to its own Tamil line among the song's lines?
    def within(V):
        hits, k0 = [], 0
        for s in P:
            n = len(s['lines']); T = vt[k0:k0 + n]; X = V[k0:k0 + n]; M = T @ X.T
            hits += list(M.argmax(1) == np.arange(n)); k0 += n
        return round(float(np.mean(hits)), 3)
    boot = lambda a: [round(float(np.percentile([rng.choice(a, len(a)).mean() for _ in range(1000)], q)), 3) for q in (2.5, 97.5)]
    cmp['m3'] = {'ta_ld': round(float(s_ld.mean()), 3), 'ta_ld_ci': boot(s_ld), 'ta_mt': round(float(s_mt.mean()), 3), 'ta_mt_ci': boot(s_mt),
                 'ld_mt': round(float(s_lm.mean()), 3), 'ld_closer': round(float(np.mean(s_ld > s_mt)), 3),
                 'align_ld': within(vl), 'align_mt': within(vm)}
    # song level (robust to line offsets in the whole-song alignment): all sampled lines of a song joined
    T = np.array([E('\n'.join(x['ta'] for x in s['lines'])) for s in P])
    Vl = np.array([E('\n'.join(x['ld'] for x in s['lines'])) for s in P]); Vm = np.array([E('\n'.join(x['mt'] for x in s['lines'])) for s in P])
    a_, b_ = (T * Vl).sum(1), (T * Vm).sum(1)
    cmp['m3_song'] = {'ta_ld': round(float(a_.mean()), 3), 'ta_mt': round(float(b_.mean()), 3), 'ld_closer': round(float(np.mean(a_ > b_)), 3),
                      'top1_ld': round(float(np.mean((T @ Vl.T).argmax(1) == np.arange(len(P)))), 3),
                      'top1_mt': round(float(np.mean((T @ Vm.T).argmax(1) == np.arange(len(P)))), 3)}
    # examples: lines where the two diverge most and the human line is closer to the Tamil
    gap = s_ld - s_mt
    ex = [dict(flat[k], sim_ld=round(float(s_ld[k]), 2), sim_mt=round(float(s_mt[k]), 2)) for k in np.argsort(-gap)[:25]]
    cmp['examples'] = ex
    out['mt'] = cmp
    print('MT', json.dumps({k: v for k, v in cmp.items() if k != 'examples'}, ensure_ascii=False)[:1500])
json.dump(out, open(os.path.join(OUT, 'translation.json'), 'w'), ensure_ascii=False, indent=1)
