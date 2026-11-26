"""Corpus statistics for LD-690 (Table 1 of the paper). Reads data/posts.json and
data/songmap/songs.json; no network. Writes paper/analysis/out/corpus_stats.json."""
import json, re, ast, os, html, collections
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'paper', 'analysis', 'out'); os.makedirs(OUT, exist_ok=True)
P = json.load(open(os.path.join(ROOT, 'data', 'posts.json')))
S = json.load(open(os.path.join(ROOT, 'data', 'songmap', 'songs.json')))
TA = re.compile(r'[஀-௿]'); TAW = re.compile(r'[஀-௿]+')
LYR = {'Vasana Kavidhai': 'Subramanya Bharathi', 'Na.Muthukumar': 'Na. Muthukumar'}
def body(p):
    c = p['content']
    if isinstance(c, str):
        try: c = ast.literal_eval(c)
        except Exception: return c
    return c['rendered'] if isinstance(c, dict) else c
def txt(s): return html.unescape(re.sub(r'<[^>]+>', '', re.sub(r'<br\s*/?>', '\n', s))).replace('\xa0', ' ')
tl = {}
for p in P:
    c = body(p)
    em = [txt(m) for m in re.findall(r'<em>(.*?)</em>', c, re.S)]
    lines = [l.strip() for e in em for l in e.split('\n') if l.strip() and not TA.search(l) and re.search(r'[A-Za-z]', l)]
    tl[str(p['id'])] = lines
par = [s for s in S if s['tamil'].strip() and any(e.strip() for e in s['english'])]
yrs = [int(str(s['year'])[:4]) for s in par if s['year']]
lyr = collections.Counter(LYR.get(s['lyricist'], s['lyricist']) for s in par)
def nl(t): return len([l for l in t.split('\n') if l.strip()])
st = {
  'posts': len(P), 'parallel_songs': len(par),
  'with_translit': sum(1 for s in par if len(tl.get(str(s['id']), [])) >= 3),
  'tamil_lines': sum(nl(s['tamil']) for s in par),
  'tamil_tokens': sum(len(TAW.findall(s['tamil'])) for s in par),
  'tamil_types': len({w for s in par for w in TAW.findall(s['tamil'])}),
  'english_lines': sum(nl(e) for s in par for e in s['english']),
  'english_tokens': sum(len(' '.join(s['english']).split()) for s in par),
  'translit_lines': sum(len(tl.get(str(s['id']), [])) for s in par),
  'dated': len(yrs), 'year_min': min(yrs), 'year_max': max(yrs),
  'post_first': min(s['date'] for s in par), 'post_last': max(s['date'] for s in par),
  'lyricists': len(lyr), 'lyricists_25plus': sum(1 for v in lyr.values() if v >= 25),
  'top_lyricists': lyr.most_common(12),
  'by_decade': sorted(collections.Counter(y // 10 * 10 for y in yrs).items()),
  'moods': collections.Counter(m for s in par for m in (s['moods'] or [])).most_common(),
  'with_mood': sum(1 for s in par if s['moods']),
}
json.dump(st, open(os.path.join(OUT, 'corpus_stats.json'), 'w'), ensure_ascii=False, indent=1)
print(json.dumps(st, ensure_ascii=False))
