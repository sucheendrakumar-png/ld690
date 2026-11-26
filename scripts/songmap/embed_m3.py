"""Meaning vectors from bge-m3, a free multilingual model that reads Tamil.

Downloads once (~590 MB, gitignored) from Hugging Face (Xenova/bge-m3, int8
ONNX). Each song is read twice: its Tamil lines and its English translation,
in chunks of up to ~400 tokens, refrains counted once. A song's vector is the
average of its Tamil and English readings, so the Tamil carries the poet's
words and the English carries the owner's reading of them.
Output: data/songmap/vectors_m3.npy (rows in vector_ids.json order)
"""
import json, os, sys, time, urllib.request
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tamil_style import tamil_doc

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SM = os.path.join(ROOT, 'data', 'songmap')
MD = os.path.join(SM, 'model', 'bge-m3')
HF = 'https://huggingface.co/Xenova/bge-m3/resolve/main/'
MAXTOK = 400

def load():
    os.makedirs(MD, exist_ok=True)
    for f, src in (('tokenizer.json', 'tokenizer.json'), ('model_quantized.onnx', 'onnx/model_quantized.onnx')):
        if not os.path.exists(os.path.join(MD, f)):
            urllib.request.urlretrieve(HF + src, os.path.join(MD, f))
    tok = Tokenizer.from_file(os.path.join(MD, 'tokenizer.json'))
    so = ort.SessionOptions(); so.intra_op_num_threads = os.cpu_count()
    sess = ort.InferenceSession(os.path.join(MD, 'model_quantized.onnx'), so, providers=['CPUExecutionProvider'])
    return tok, sess

def chunks(lines, tok):
    out, cur, n = [], [], 0
    for l in lines:
        k = len(tok.encode(l, add_special_tokens=False).ids)
        if cur and n + k > MAXTOK:
            out.append('\n'.join(cur)); cur, n = [], 0
        cur.append(l); n += k
    if cur:
        out.append('\n'.join(cur))
    return out

def embed(tok, sess, text):
    e = tok.encode(text)
    ids = np.array([e.ids[:512]], dtype=np.int64); am = np.ones_like(ids)
    names = {i.name for i in sess.get_inputs()}
    feeds = {'input_ids': ids, 'attention_mask': am}
    if 'token_type_ids' in names:
        feeds['token_type_ids'] = np.zeros_like(ids)
    h = sess.run(None, {k: v for k, v in feeds.items() if k in names})[0]
    v = h[0, 0] if h.ndim == 3 else h[0]   # CLS pooling
    return v / np.linalg.norm(v)

def uniq(lines):
    seen, out = set(), []
    for l in lines:
        l = l.strip(); k = l.lower()
        if l and k not in seen:
            seen.add(k); out.append(l)
    return out

def main():
    S = {s['id']: s for s in json.load(open(os.path.join(SM, 'songs.json')))}
    ids = json.load(open(os.path.join(SM, 'vector_ids.json')))
    tok, sess = load()
    T, E = [], []
    t0 = time.time()
    for n, i in enumerate(ids):
        s = S[i]
        ta = chunks(uniq(tamil_doc(s['tamil']).split('\n')), tok)
        en = chunks(uniq([l for b in s['english'] for l in b.split('\n')]), tok)
        tv = np.mean([embed(tok, sess, c) for c in ta], 0) if ta else None
        ev = np.mean([embed(tok, sess, c) for c in en], 0) if en else None
        tv = tv if tv is not None else ev; ev = ev if ev is not None else tv
        T.append(tv / np.linalg.norm(tv)); E.append(ev / np.linalg.norm(ev))
        if n % 50 == 0:
            print(n, round(time.time() - t0), 's', flush=True)
    T, E = np.array(T), np.array(E)
    M = T + E; M /= np.linalg.norm(M, axis=1, keepdims=True)
    np.save(os.path.join(SM, 'vectors_m3_ta.npy'), T.astype(np.float32))
    np.save(os.path.join(SM, 'vectors_m3_en.npy'), E.astype(np.float32))
    np.save(os.path.join(SM, 'vectors_m3.npy'), M.astype(np.float32))
    print('done', M.shape, round(time.time() - t0), 's')

if __name__ == '__main__':
    main()
