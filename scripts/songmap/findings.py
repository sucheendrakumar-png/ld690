import json,re,collections,math,ast,numpy as np
S=json.load(open('data/songmap/songs.json'))
LYR={'Vasana Kavidhai':'Subramanya Bharathi','Na.Muthukumar':'Na. Muthukumar'}
for s in S:
    s['lyr']=LYR.get(s['lyricist'],s['lyricist'])
    try: s['y']=int(str(s['year'])[:4])
    except: s['y']=None
    s['v']=int(s['views'] or 0)
    s['words']=re.findall(r'[஀-௿]+',s['tamil'])
    lines=[]; seen=set()
    for l in s['tamil'].split('\n'):
        l=l.strip()
        if l and l not in seen: seen.add(l); lines.append(l)
    s['uw']=re.findall(r'[஀-௿]+',' '.join(lines))
    s['rep']=1-len(seen)/max(1,len([l for l in s['tamil'].split('\n') if l.strip()]))
def dec(y): return None if not y else (y//10*10 if y>=1950 else 1940)
film=[s for s in S if s['y'] and s['y']>=1950 and 'Bharathi' not in s['lyr'] and 'Siddhar' not in s['lyr']]
print('n film-dated',len(film))
G=set('ஜஷஸஹஶ')
print('\n# grantha letters per 1000 letters, by decade; repetition; words/song; unique-word ratio')
for d in sorted({dec(s['y']) for s in film}):
    ss=[s for s in film if dec(s['y'])==d]
    txt=''.join(''.join(s['uw']) for s in ss)
    g=sum(c in G for c in txt)/len(txt)*1000
    rep=np.mean([s['rep'] for s in ss])
    nw=np.median([len(s['uw']) for s in ss])
    wl=np.mean([len(w) for s in ss for w in s['uw']])
    print(d,len(ss),'grantha %.1f'%g,'rep %.2f'%rep,'words %d'%nw,'wordlen %.2f'%wl)
# word trends
print('\n# share of songs containing word, by era')
eras=[(1950,1979),(1980,1999),(2000,2026)]
W=['காதல்','அம்மா','கண்ணே','தமிழ்','நிலா','வானம்','மழை','கடவுள்','இறைவ','உயிர்','நெஞ்சம்','மனசு','மனது','இதயம்','விதி','பெண்','கனவு','மரணம்','பணம்','உலகம்','தாய்','நண்ப','சிரி','கண்ணீர்','காற்று','போதை','நீ']
for w in W:
    row=[]
    for a,b in eras:
        ss=[s for s in film if a<=s['y']<=b]
        row.append(sum(any(x.startswith(w) for x in s['uw']) for s in ss)/len(ss))
    print(w,' '.join('%.0f%%'%(r*100) for r in row))
print([len([s for s in film if a<=s['y']<=b]) for a,b in eras])
# distinctive words per lyricist (log-odds with prior)
cnt=collections.Counter(s['lyr'] for s in S)
tot=collections.Counter(); per=collections.defaultdict(collections.Counter)
for s in S:
    st=set(s['uw'])
    for w in st: tot[w]+=1; per[s['lyr']][w]+=1
N=len(S)
print('\n# distinctive words (share of poet songs vs others)')
for l,n in cnt.most_common(8):
    sc=[]
    for w,c in per[l].items():
        if c<max(4,n*0.12): continue
        o=(tot[w]-c)/(N-n)
        sc.append(((c/n)/(o+0.01),w,c/n,o))
    sc.sort(reverse=True)
    print(l,n,' | '.join('%s %.0f%%/%.0f%%'%(w,a*100,b*100) for _,w,a,b in sc[:8]))
# vocabulary richness: unique words in random 300-word samples
print('\n# vocabulary richness (distinct words per 1000 tokens, 20 resamples)')
rng=np.random.default_rng(0)
for l,n in cnt.most_common(10):
    toks=[w for s in S if s['lyr']==l for w in s['uw']]
    if len(toks)<1000: continue
    r=np.mean([len(set(rng.choice(toks,1000,replace=False))) for _ in range(20)])
    print(l,n,int(r))
# views per poet
print('\n# median and mean views per song by poet (12 months)')
for l,n in cnt.most_common(10):
    v=[s['v'] for s in S if s['lyr']==l]
    print(l,n,int(np.median(v)),int(np.mean(v)),sum(v))
print('\n# views by decade')
for d in sorted({dec(s['y']) for s in S if s['y']}):
    v=[s['v'] for s in S if s['y'] and dec(s['y'])==d]
    print(d,len(v),int(np.median(v)),int(np.mean(v)))
# moods by era
print('\n# moods by era')
for a,b in eras:
    ss=[s for s in film if a<=s['y']<=b]
    c=collections.Counter(m for s in ss for m in ast.literal_eval(s['moods']) if isinstance(s['moods'],str)) if isinstance(ss[0]['moods'],str) else collections.Counter(m for s in ss for m in s['moods'])
    print(a,b,len(ss),{k:'%.0f%%'%(v/len(ss)*100) for k,v in c.most_common()})
