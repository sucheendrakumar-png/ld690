"""Parse data/posts.json into one clean record per song (no network).

Output: data/songmap/songs.json with title, film, song, link, lyricist,
year, singer, composer, moods, views, and the English translation text
split into stanzas (what gets embedded).
"""
import json, re, html, os, sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = lambda *p: os.path.join(ROOT, 'data', *p)

TAMIL = re.compile(r'[஀-௿]')
TAG = re.compile(r'<[^>]+>')

def text(s):
    s = re.sub(r'<br\s*/?>', '\n', s)
    s = TAG.sub('', s)
    return html.unescape(s).replace('\xa0', ' ').strip()

STOP = set("the a an and of to in is my your you i me it that this for on with be are was will not no what who how when where why all his her he she we our they their from by as at so but if or love heart life".split())

def english_score(s):
    w = re.findall(r"[A-Za-z']+", s.lower())
    return (sum(1 for x in w if x in STOP) / len(w)) if len(w) >= 4 else 0

def english_blocks(c):
    out = []
    # Gutenberg: any element in the translation colour (paragraph or inline span)
    for m in re.finditer(r'<(p|span)\b[^>]*has-vivid-cyan-blue-color[^>]*>(.*?)</\1>', c, re.S):
        out.append(text(m.group(2)))
    if out:
        return out
    # Classic editor: teal/blue spans
    spans = re.findall(r'<span[^>]*style="[^"]*color:\s*#[0-9a-fA-F]{6}[^"]*"[^>]*>(.*?)</span>', c, re.S | re.I)
    spans = [text(x) for x in spans if not TAMIL.search(x)]
    spans = [x for x in spans if english_score(x) > 0.15]
    if spans:
        return ['\n'.join(spans)]
    # Fallback: block-level text that reads as English (skip intro, metadata, Tamil, transliteration)
    blocks = re.findall(r'<(pre|p)\b[^>]*>(.*?)</\1>', c, re.S)
    out = []
    for i, (_, b) in enumerate(blocks):
        t = text(b)
        if i == 0 or TAMIL.search(t) or re.match(r'(Song|Film|Movie)\s*:', t):
            continue
        lines = [l for l in t.split('\n') if english_score(l + ' x x x') > 0.2 or english_score(l) > 0.2]
        if lines and english_score(t) > 0.2:
            out.append('\n'.join(lines))
    return out

# The owner's own notes sometimes share the translation colour (praise for the
# singer, thank-yous, hashtags). They are not the song, so they are dropped.
NOTE = re.compile(r"#\w|thanks? (to|for)|suggestion|lyrics by|sung by|rendition|penned|composed by|"
                  r"music by|this song|the song|song from|from the movie|in the movie|masterpiece|"
                  r"brilliant|legendary|goosebumps|enjoy!|happy valentine|kudos", re.I)

def clean_english(block):
    return '\n'.join(l for l in block.split('\n') if not NOTE.search(l) and len(l) < 260)

def tamil_text(c):
    lines = [l for l in text(c).split('\n') if TAMIL.search(l)]
    return '\n'.join(lines)

META_KEYS = {'song': 'song', 'film': 'film', 'movie': 'film', 'source': 'film', 'album': 'film',
             'year': 'year', 'singer': 'singer', 'singers': 'singer', 'sung by': 'singer', 'vocals': 'singer',
             'composer': 'composer', 'music': 'composer', 'music director': 'composer',
             'lyricist': 'lyricist', 'lyrics': 'lyricist', 'poet': 'lyricist'}

def metadata(c):
    meta = {}
    for m in re.finditer(r'<p[^>]*>(.*?)</p>', c, re.S):
        body = m.group(1)
        if '<br' not in body or ':' not in body:
            continue
        got = {}
        for line in text(body).split('\n'):
            if ':' not in line:
                continue
            k, v = line.split(':', 1)
            k = k.strip().lower()
            if k in META_KEYS and v.strip():
                got[META_KEYS[k]] = v.strip()
        if len(got) >= 2:
            meta = got
            break
    return meta

def main():
    posts = json.load(open(D('posts.json')))
    cats = {c['id']: html.unescape(c['name']) for c in json.load(open(D('categories.json')))}
    tags = {t['id']: html.unescape(t['name']) for t in json.load(open(D('tags.json')))}
    moods = defaultdict(list)
    ids_path = D('collections', 'ids.json')
    if os.path.exists(ids_path):
        for k, v in json.load(open(ids_path))['all'].items():
            for pid in v:
                moods[int(pid)].append(k)
    views = defaultdict(int)
    tp = D('topposts-2026-11-24.json')
    if os.path.exists(tp):
        for day in json.load(open(tp))['days'].values():
            for r in day.get('postviews', []):
                if r.get('id'):
                    views[int(r['id'])] += r.get('views', 0)
    songs = []
    for p in posts:
        if p.get('status') != 'publish':
            continue
        c = p['content']['rendered'] if isinstance(p['content'], dict) else p['content']
        title = html.unescape(p['title']['rendered'] if isinstance(p['title'], dict) else p['title']).strip()
        title = re.sub(r'\s*\.{2,}\s*$', '', title)
        film, song = '', title
        m = re.match(r'\s*\[([^\]]+)\]\s*(.*)', title)
        if m:
            film, song = m.group(1).strip(), m.group(2).strip()
        meta = metadata(c)
        eng = [clean_english(b) for b in english_blocks(c)]
        eng = [b for b in eng if len(b) > 3]
        cat_ids = p['categories'] if isinstance(p['categories'], list) else json.loads(p['categories'])
        tag_ids = p['tags'] if isinstance(p['tags'], list) else json.loads(p['tags'])
        lyr = [cats.get(i, '') for i in cat_ids if cats.get(i) not in (None, 'Uncategorized', 'blog', 'Ai News')]
        year = meta.get('year', '')
        ym = re.search(r'(19|20)\d\d', year)
        if not ym:
            for t in tag_ids:
                n = tags.get(t, '')
                if re.fullmatch(r'(19|20)\d\d', n.strip()):
                    ym = re.match(r'.*', n.strip()); break
        songs.append({
            'id': int(p['id']), 'link': p['link'], 'date': p['date'][:10],
            'title': title, 'film': meta.get('film') or film, 'song': meta.get('song') or song,
            'lyricist': lyr[0] if lyr else meta.get('lyricist', ''),
            'year': int(ym.group(0)) if ym else None,
            'singer': meta.get('singer', ''), 'composer': meta.get('composer', ''),
            'moods': moods.get(int(p['id']), []), 'views': views.get(int(p['id']), 0),
            'english': eng, 'tamil': tamil_text(c),
        })
    os.makedirs(D('songmap'), exist_ok=True)
    json.dump(songs, open(D('songmap', 'songs.json'), 'w'), ensure_ascii=False, indent=0)
    n = len(songs)
    print(n, 'songs;',
          'english', sum(1 for s in songs if s['english']),
          'year', sum(1 for s in songs if s['year']),
          'composer', sum(1 for s in songs if s['composer']),
          'singer', sum(1 for s in songs if s['singer']),
          'moods', sum(1 for s in songs if s['moods']))
    for s in songs:
        if not s['english']:
            print('  no english:', s['id'], s['title'])

if __name__ == '__main__':
    main()
