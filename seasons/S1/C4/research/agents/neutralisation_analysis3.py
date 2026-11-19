"""LOOK-AHEAD ANALYSIS: in rows where T != zx(fwd sum), what explains the gap?"""
import sys; sys.path.insert(0,'.')
import pandas as pd, numpy as np
from lab.paths import data_root
from lab.labels import window_sum
from lab.metrics import row_cos
from lab.honest import zx, rank, blocks, residualize
r=data_root(); rows=[]
for p in ['007','008','010','022','025']:
    X=pd.read_parquet(r/f'{p}.small.returns.parquet'); T=pd.read_parquet(r/f'{p}.small.target.parquet'); F=pd.read_parquet(r/f'{p}.small.features.parquet')
    n=len(X); a=n-6000; X=X.iloc[a-400:n-200]; T=T.loc[X.index]; F=F.loc[X.index]
    Z=zx(window_sum(X,120,1)); c=row_cos(Z,T)
    bad=c[(c<0.999)&c.notna()].index; bad=bad[bad>=X.index[400]]
    G=residualize(T.loc[bad],[Z.loc[bad]])
    g=blocks(F)
    regs={f'rF{i}':rank(g[i]) for i in g}
    regs.update({f'rdF{i}_120':rank(g[i].diff(120)) for i in (1,5,6)})
    for k in (1,24,120): regs[f'past{k}']=zx(X.rolling(k).sum())
    regs['fwd1']=zx(X.shift(-1)); regs['fwd121']=zx(X.shift(-121)); regs['ret0']=zx(X)
    regs['fwdsum_off0']=zx(window_sum(X,120,0)); regs['fwdsum_off2']=zx(window_sum(X,120,2))
    regs['fwd_h96']=zx(window_sum(X,96,1)); regs['fwd_h144']=zx(window_sum(X,144,1))
    for k,v in regs.items():
        rows.append((p,k,float(row_cos(G,residualize(v.loc[bad],[Z.loc[bad]])).mean())))
    # characterise: does gap look like whole-ticker shifts? per-ticker mean of gap
    print(p,'bad hours',len(bad),'mean cos in bad',float(c.loc[bad].mean()),'gap ticker-mean share', float((G.mean()**2).sum()/ (G**2).mean().sum()))
    # are bad hours contiguous?
    b=pd.Series(1,index=bad); print('  runs: first/last bad',bad.min()-a,bad.max()-a)
print(pd.DataFrame(rows,columns=['p','reg','cos_gap']).pivot(index='reg',columns='p',values='cos_gap').round(3))
