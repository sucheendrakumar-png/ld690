"""Machine-translation baseline for the LD-690 translations.

Aligns Tamil and English lines (matched stanza pairs from the post, or the whole
song when its line counts agree), samples up to PER_POET songs for each of the seven
lyricists with 25+ songs (first MAX_LINES distinct lines each), and translates every Tamil
line with NLLB-200 (distilled 600M, full-precision ONNX, greedy decoding). Each line gets a
final full stop: without it, NLLB answers short verse fragments with invented Bible quotations.

Model files (not redistributed), fetched once from the Hugging Face hub:
    Xenova/nllb-200-distilled-600M: tokenizer.json, onnx/encoder_model.onnx, onnx/decoder_model.onnx,
    onnx/decoder_with_past_model.onnx  ->  paper/ext/nllb/
Writes paper/ext/mt_pairs.json (Tamil line, LD-690 English line, NLLB English line).
"""
import json, os, re, html, random, sys, time
from collections import defaultdict
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXT = os.path.join(ROOT, 'paper', 'ext')
M = os.path.join(EXT, 'nllb')
TAMIL = re.compile(r'[஀-௿]')
PER_POET = int(os.environ.get('PER_POET', 10))
MAX_LINES = int(os.environ.get('MAX_LINES', 12))
MERGE = {'Vasana Kavidhai': 'Subramanya Bharathi'}

def text(s):
    s = re.sub(r'<br\s*/?>', '\n', s)
    s = re.sub(r'<[^>]+>', '', s)
    return html.unescape(s).replace('\xa0', ' ').strip()

def lines(t):
    return [l.strip() for l in t.split('\n') if l.strip()]

def stanza_pairs(content):
    """Yield (tamil_lines, english_lines) for each quote followed by a translation paragraph."""
    blocks = re.finditer(r'<blockquote[^>]*>(.*?)</blockquote>|<p\b([^>]*)>(.*?)</p>', content, re.S)
    ta = None; seen = set()
    for b in blocks:
        if b.group(1) is not None:
            t = [l for l in lines(text(b.group(1))) if TAMIL.search(l)]
            ta = t if t and TAMIL.search(' '.join(t)) else None
        elif 'has-vivid-cyan-blue-color' in (b.group(2) or '') + (b.group(3) or '') and ta:
            en = lines(text(b.group(3)))
            if len(en) == len(ta) and not TAMIL.search(' '.join(en)) and tuple(ta) not in seen:
                seen.add(tuple(ta))   # refrains once
                yield ta, en
            ta = None

class NLLB:
    """Greedy decoding with a key/value cache (first step: decoder_model, then decoder_with_past)."""
    def __init__(self):
        o = ort.SessionOptions(); o.intra_op_num_threads = int(os.environ.get('MT_THREADS', os.cpu_count()))
        P = lambda f: os.path.join(M, 'onnx', f + '.onnx')
        q = os.environ.get('MT_PREC', '')   # '' = full precision, '_quantized' = int8
        self.enc = ort.InferenceSession(P('encoder_model' + q), o)
        self.dec0 = ort.InferenceSession(P('decoder_model' + q), o)
        self.dec = ort.InferenceSession(P('decoder_with_past_model' + q), o)
        self.out0 = [x.name for x in self.dec0.get_outputs()]
        self.out1 = [x.name for x in self.dec.get_outputs()]
        self.tok = Tokenizer.from_file(os.path.join(M, 'tokenizer.json'))
        self.src = self.tok.token_to_id('tam_Taml'); self.tgt = self.tok.token_to_id('eng_Latn')
        self.eos = self.tok.token_to_id('</s>')

    def __call__(self, s, max_len=60):
        ids = self.tok.encode(s, add_special_tokens=False).ids[:120]
        x = np.array([[self.src] + ids + [self.eos]], dtype=np.int64)
        mask = np.ones_like(x)
        h = self.enc.run(None, {'input_ids': x, 'attention_mask': mask})[0]
        # NLLB decoding starts with </s> followed by the forced target-language token
        r = self.dec0.run(None, {'input_ids': np.array([[self.eos, self.tgt]], dtype=np.int64),
                                 'encoder_hidden_states': h, 'encoder_attention_mask': mask})
        res = dict(zip(self.out0, r))
        past = {k.replace('present', 'past_key_values'): v for k, v in res.items() if k.startswith('present')}
        out = []
        nxt = int(res['logits'][0, -1].argmax())
        for _ in range(max_len):
            if nxt == self.eos:
                break
            out.append(nxt)
            if len(out) > 6 and len(set(out[-6:])) == 1:   # stop runaway repetition
                break
            feed = {'input_ids': np.array([[nxt]], dtype=np.int64), 'encoder_attention_mask': mask}
            feed.update(past)
            r = dict(zip(self.out1, self.dec.run(None, feed)))
            for k, v in r.items():
                if k.startswith('present'):
                    past[k.replace('present', 'past_key_values')] = v   # decoder self-attention grows; encoder kv kept
            nxt = int(r['logits'][0, -1].argmax())
        return self.tok.decode(out, skip_special_tokens=True).strip()

def aligned(song, content):
    """Line-aligned (Tamil, English) pairs: matched stanza pairs from the post, else the whole song
    when its Tamil and English line counts agree. Refrains once, first MAX_LINES lines."""
    pairs = [p for t, e in stanza_pairs(content) for p in zip(t, e)]
    if len(pairs) < 6:
        ta = lines(song['tamil']); en = [l for b in song['english'] for l in lines(b)]
        pairs = list(zip(ta, en)) if len(ta) == len(en) else []
    seen, out = set(), []
    for t, e in pairs:
        if t not in seen:
            seen.add(t); out.append((t, e))
    return out[:MAX_LINES]

def main():
    posts = {p['id']: p for p in json.load(open(os.path.join(ROOT, 'data', 'posts.json')))}
    songs = [s for s in json.load(open(os.path.join(ROOT, 'data', 'songmap', 'songs.json'))) if s['english'] and s['tamil'].strip()]
    cnt = defaultdict(int)
    for s in songs: cnt[MERGE.get(s['lyricist'], s['lyricist'])] += 1
    big = sorted(l for l, n in cnt.items() if n >= 25)
    by = defaultdict(list)
    for s in songs:
        l = MERGE.get(s['lyricist'], s['lyricist'])
        if l in big:
            a = aligned(s, posts[s['id']]['content']['rendered'])
            if len(a) >= 6:
                by[l].append((s, a))
    rng = random.Random(7)
    sample = []
    for l in big:
        pool = sorted(by[l], key=lambda x: x[0]['id']); rng.shuffle(pool)
        sample += [(l, s, a) for s, a in pool[:PER_POET]]
    mt = NLLB(); cache = {}; t0 = time.time(); out = []
    todo = sum(len(a) for _, _, a in sample); done = 0
    for l, s, a in sample:
        rec = {'id': s['id'], 'lyricist': l, 'year': s['year'], 'moods': s['moods'], 'lines': []}
        for t, e in a:
            if t not in cache:
                cache[t] = mt(t if t.rstrip()[-1:] in '.?!' else t.rstrip() + '.')   # a final stop avoids NLLB's hallucinations on bare fragments
            rec['lines'].append({'ta': t, 'ld': e, 'mt': cache[t]}); done += 1
        out.append(rec)
        print(f'{len(out)}/{len(sample)} songs, {done}/{todo} lines, {time.time() - t0:.0f}s', flush=True)
        json.dump(out, open(os.path.join(EXT, 'mt_pairs.json'), 'w'), ensure_ascii=False, indent=0)

if __name__ == '__main__':
    if sys.argv[1:] == ['test']:
        mt = NLLB()
        for s in ['வேல் விழி போடும் தூண்டிலே', 'உருகி உருகி போனதடி', 'கரைகளில் கரையும் வெண்ணுறை']:
            t = time.time(); print(s, '->', mt(s), f'{time.time() - t:.2f}s')
    else:
        main()
