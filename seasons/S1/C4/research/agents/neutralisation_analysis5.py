"""LOOK-AHEAD ANALYSIS: recover the 'true' demeaned returns implied by sigma*T (backward recursion from the clean
recent region) and characterise eps = c(X) - c(Xtrue) in old rows."""
import sys; sys.path.insert(0,'.')
import pandas as pd, numpy as np
from lab.paths import data_root
from lab.honest import zx, rank, blocks
from lab.metrics import row_cos
r=data_root()
L=lambda p,k: pd.read_parquet(r/f'{p}.small.{k}.parquet')
for p in ['008','010','022']:
    X,T,S,F=L(p,'returns'),L(p,'target'),L(p,'dispersion').iloc[:,0],L(p,'features')
    n=len(X); X=X.iloc[-7000:]; T=T.loc[X.index]; S=S.loc[X.index]; F=F.loc[X.index]
    cX=X.sub(X.mean(axis=1),axis=0).to_numpy(); Dm=(T.mul(S,axis=0)).to_numpy()
    m=len(X); xt=cX.copy()
    start=m-400   # clean region seed: rows >= start-? keep X
    for t in range(start-121, -1, -1):   # c(x_{t+1}) = D_t - D_{t+1} + c(x_{t+121})
        xt[t+1]=Dm[t]-Dm[t+1]+xt[t+121]
    eps=cX-xt
    E=pd.DataFrame(eps,index=X.index,columns=X.columns)
    mag=np.sqrt((eps**2).sum(1))/np.sqrt((cX**2).sum(1))
    seg=pd.Series(mag,index=X.index)
    print(p,'rel |eps| by 1000-row block (old->new):',[round(float(seg.iloc[i:i+1000].median()),3) for i in range(0,7000,1000)])
    cs=row_cos(pd.DataFrame(xt,index=X.index,columns=X.columns),pd.DataFrame(cX,index=X.index,columns=X.columns))
    print('   cos(Xtrue,X) by block',[round(float(cs.iloc[i:i+1000].mean()),3) for i in range(0,7000,1000)])
    g=blocks(F); old=X.index[200:4000]
    for name,v in [('rF1',rank(g[1])),('rF5',rank(g[5])),('rF6',rank(g[6])),('past1',zx(X)),('lag1',zx(X.shift(1)))]:
        print('   cos(eps,',name,') old', round(float(row_cos(E.loc[old],v.loc[old]).mean()),4))
    # eps autocorr
    print('   eps autocorr lag1', round(float(row_cos(E.loc[old],E.shift(1).loc[old]).mean()),3), 'lag24', round(float(row_cos(E.loc[old],E.shift(24).loc[old]).mean()),3))
