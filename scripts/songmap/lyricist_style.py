"""Lyricist style compass: can the Tamil alone tell who wrote a song?

A linear classifier (LinearSVC) learns the seven lyricists with 25+ songs from
short letter runs (1-3 Tamil characters, inside words) of their lyrics. Every
song's position comes from a model that never saw that song: songs by the
seven are scored with 10-fold cross-validation, everyone else's songs with
the model trained on all of the seven. So the clusters are earned, not drawn.

Layout: the seven lyricists sit around a circle (similar styles next to each
other, from the confusion matrix). Each song sits at the average of the
lyricists' positions, weighted by how much its Tamil resembles each one.
A song deep in a corner is unmistakably that poet; one near the middle is
written in a voice several poets share.
Output: data/songmap/style.json
"""
import json, os, sys, itertools
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, balanced_accuracy_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tamil_style import tamil_doc

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SM = os.path.join(ROOT, 'data', 'songmap')
MERGE = {'Vasana Kavidhai': 'Subramanya Bharathi'}
MIN = 25
TEMP = 0.12  # softmax temperature on the SVM scores (lower = songs pulled harder to their best match)

def model():
    return make_pipeline(TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 3), min_df=2, sublinear_tf=True),
                         LinearSVC(C=0.3, class_weight='balanced'))

def softmax(z):
    z = z / TEMP
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)

def main():
    S = {s['id']: s for s in json.load(open(os.path.join(SM, 'songs.json')))}
    ids = json.load(open(os.path.join(SM, 'vector_ids.json')))
    lab = np.array([MERGE.get(S[i]['lyricist'], S[i]['lyricist']) for i in ids])
    docs = np.array([tamil_doc(S[i]['tamil']) for i in ids], dtype=object)
    cnt = Counter(lab)
    big = sorted([l for l in cnt if cnt[l] >= MIN], key=lambda l: -cnt[l])
    inb = np.isin(lab, big)
    cv = StratifiedKFold(10, shuffle=True, random_state=7)
    m = model()
    oof = cross_val_predict(m, list(docs[inb]), lab[inb], cv=cv, method='decision_function')
    classes = [str(c) for c in sorted(big)]  # sklearn orders classes alphabetically
    pred = np.array(classes)[oof.argmax(1)]
    bacc = balanced_accuracy_score(lab[inb], pred)
    cm = confusion_matrix(lab[inb], pred, labels=classes, normalize='true')
    m.fit(list(docs[inb]), lab[inb])
    rest = m.decision_function(list(docs[~inb]))
    scores = np.zeros((len(ids), len(classes)))
    scores[inb] = oof; scores[~inb] = rest
    P = softmax(scores)
    # order lyricists around the circle so that often-confused pairs are neighbours
    conf = (cm + cm.T) / 2
    best, bo = -1, None
    for perm in itertools.permutations(range(1, len(classes))):
        order = (0,) + perm
        sc = sum(conf[order[i], order[(i + 1) % len(order)]] for i in range(len(order)))
        if sc > best:
            best, bo = sc, order
    ang = {classes[k]: np.pi / 2 + 2 * np.pi * i / len(bo) for i, k in enumerate(bo)}
    A = np.array([[np.cos(ang[c]), np.sin(ang[c])] for c in classes])
    xy = P @ A
    rng = np.random.default_rng(7)
    xy += rng.normal(0, 0.045, xy.shape)  # small fixed jitter so overlapping songs stay clickable
    xy = (xy + 1.15) / 2.3
    recall = {c: round(float(cm[i, i]), 3) for i, c in enumerate(classes)}
    # the most confused pairs (both directions summed)
    pairs = sorted(((conf[i, j], classes[i], classes[j]) for i in range(len(classes)) for j in range(i + 1, len(classes))), reverse=True)[:3]
    out = {
        'lyricists': [{'name': c, 'n': int(cnt[c]), 'recall': recall[c],
                       'x': round(float((A[i, 0] * 1.0 + 1.15) / 2.3), 4), 'y': round(float((A[i, 1] * 1.0 + 1.15) / 2.3), 4)}
                      for i, c in enumerate(classes)],
        'accuracy': round(float(bacc), 3), 'chance': round(1 / len(classes), 3), 'tested': int(inb.sum()),
        'classes': classes, 'confusion': [[round(float(v), 3) for v in row] for row in cm],
        'confused': [{'a': a, 'b': b, 'rate': round(float(r), 3)} for r, a, b in pairs],
        'points': {str(ids[i]): {'sx': round(float(xy[i, 0]), 4), 'sy': round(float(xy[i, 1]), 4),
                                 'guess': classes[int(scores[i].argmax())], 'sure': round(float(P[i].max()), 2),
                                 'held_out': bool(inb[i])} for i in range(len(ids))},
    }
    json.dump(out, open(os.path.join(SM, 'style.json'), 'w'), ensure_ascii=False)
    print('balanced accuracy', round(bacc, 3), 'chance', round(1 / len(classes), 3), 'songs', int(inb.sum()))
    print('recall', recall)
    print('order', [classes[k] for k in bo])
    print('confused', out['confused'])
    for i, c in enumerate(classes):
        print(f'{c:22s}', ' '.join(f'{v:4.2f}' for v in cm[i]))

if __name__ == '__main__':
    main()
