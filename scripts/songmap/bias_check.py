exec(open(__import__("os").path.join(__import__("os").path.dirname(__file__),"findings.py")).read().split("def dec(y)")[0])
import ast
def moods(s): return ast.literal_eval(s['moods']) if isinstance(s['moods'],str) else s['moods']
film=[s for s in S if s['y'] and s['y']>=1950 and 'Bharathi' not in s['lyr'] and 'Siddhar' not in s['lyr']]
eras=[(1950,1979),(1980,1999),(2000,2026)]
def share(ss,w): return '%d%% (n=%d)'%(100*sum(any(x.startswith(w) for x in s['uw']) for s in ss)/max(1,len(ss)),len(ss)) if ss else '-'
def row(lbl,f,w):
    print(lbl,w,' | '.join(share([s for s in film if a<=s['y']<=b and f(s)],w) for a,b in eras))
for w in ['காதல்','மழை','கனவு','இறைவ','நீ']:
    row('ALL',lambda s:True,w)
    row('love-mood only',lambda s:'love' in moods(s),w)
    row('philosophy only',lambda s:'philosophy' in moods(s),w)
    row('no Kannadasan',lambda s:s['lyr']!='Kannadasan',w)
    print()
# within poets across their own careers
print('# within poet, early vs late career')
for l in ['Vairamuthu','Vaali','Kannadasan']:
    ss=sorted([s for s in film if s['lyr']==l],key=lambda s:s['y'])
    h=len(ss)//2; A,B=ss[:h],ss[h:]
    def wl(x): return np.mean([len(w) for s in x for w in s['uw']])
    def sh(x,w): return 100*sum(any(t.startswith(w) for t in s['uw']) for s in x)/len(x)
    print(l,'%d-%d vs %d-%d'%(A[0]['y'],A[-1]['y'],B[0]['y'],B[-1]['y']),'wordlen %.2f -> %.2f'%(wl(A),wl(B)),'kaadhal %.0f->%.0f'%(sh(A,'காதல்'),sh(B,'காதல்')),'nee %.0f->%.0f'%(sh(A,'நீ'),sh(B,'நீ')),'mazhai %.0f->%.0f'%(sh(A,'மழை'),sh(B,'மழை')))
# word length by decade within love-mood songs
print('# word length by era, love songs only / philosophy only')
for a,b in eras:
    for m in ['love','philosophy']:
        ss=[s for s in film if a<=s['y']<=b and m in moods(s)]
        print(a,m,len(ss),'%.2f'%np.mean([len(w) for s in ss for w in s['uw']]))
