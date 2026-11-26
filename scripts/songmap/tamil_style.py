"""Tamil wording/style vectors: what the Tamil lyrics actually say and how.

No model download. Each song's Tamil lines (refrains counted once) become
TF-IDF features over (a) whole Tamil words and word pairs, which carry
vocabulary and imagery, and (b) letter sequences of 2-4 Tamil characters
inside words, which carry suffixes, verb endings, archaic vs colloquial
forms and sandhi, i.e. style. Both are reduced with SVD to 120 dimensions.
Output: data/songmap/vectors_ta.npy (same row order as vector_ids.json)
"""
import json, os, re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SM = os.path.join(ROOT, 'data', 'songmap')
WORD = re.compile(r'[஀-௿]+')

def tamil_doc(t):
    seen, out = set(), []
    for l in t.split('\n'):
        w = WORD.findall(l)
        k = ' '.join(w)
        if w and k not in seen:
            seen.add(k); out.append(k)
    return '\n'.join(out)

def main():
    S = {s['id']: s for s in json.load(open(os.path.join(SM, 'songs.json')))}
    ids = json.load(open(os.path.join(SM, 'vector_ids.json')))
    docs = [tamil_doc(S[i]['tamil']) for i in ids]
    wv = TfidfVectorizer(token_pattern=r'[஀-௿]+', ngram_range=(1, 2), min_df=2, max_df=0.5, sublinear_tf=True)
    cv = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=3, max_df=0.6, sublinear_tf=True)
    Xw = wv.fit_transform(docs); Xc = cv.fit_transform(docs)
    Zw = TruncatedSVD(80, random_state=7).fit_transform(Xw)
    Zc = TruncatedSVD(80, random_state=7).fit_transform(Xc)
    Z = np.hstack([normalize(Zw), normalize(Zc)])
    Z = normalize(Z - Z.mean(0))
    empty = [ids[i] for i, d in enumerate(docs) if len(d) < 20]
    np.save(os.path.join(SM, 'vectors_ta.npy'), Z.astype(np.float32))
    print('tamil vectors', Z.shape, 'features', Xw.shape[1], Xc.shape[1], 'too little Tamil:', empty)

if __name__ == '__main__':
    main()
