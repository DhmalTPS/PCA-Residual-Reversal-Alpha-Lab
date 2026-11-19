"""NEUTRALISATION family - LOOK-AHEAD ANALYSIS ONLY (never used inside fn).
Which transform of the forward 120h window sum (offset 1) reproduces T?  Test hypotheses by mean per-hour cosine."""
import sys; sys.path.insert(0,'.')
import pandas as pd, numpy as np
from lab.paths import data_root
from lab.labels import window_sum
from lab.metrics import row_cos
from lab.honest import zx, rank
r=data_root()
def cs(df): return df.sub(df.mean(axis=1),axis=0)
res=[]
for p in ['005','015','025','029']:
    X=pd.read_parquet(r/f'{p}.small.returns.parquet'); T=pd.read_parquet(r/f'{p}.small.target.parquet'); D=pd.read_parquet(r/f'{p}.small.dispersion.parquet').iloc[:,0]
    n=len(X); sl=slice(n-3000,n-130)
    X=X.iloc[n-3000-1500:]; T=T.loc[X.index]; D=D.loc[X.index]
    idx=X.index[1500:-130]
    m=X.mean(axis=1)
    print(p,'X rowmean/xsstd', float(m.abs().mean()/X.std(axis=1).mean()))
    W=lambda Y: window_sum(Y,120,1).loc[idx]
    Tv=T.loc[idx]
    H={}
    H['sum']=W(X)
    H['sum_xsdemean']=W(cs(X))
    vol=X.rolling(720).std()
    H['sum_X/vol720']=W(X/vol)
    H['sum_X/vol168']=W(X/X.rolling(168).std())
    # beta-neutral residual
    beta=X.rolling(720).cov(m) .div(m.rolling(720).var(),axis=0)
    H['sum_beta_resid']=W(X.sub(beta.mul(m,axis=0)))
    H['logsum']=W(np.log1p(X))
    H['prod']=np.exp(W(np.log1p(X)))-1
    for k,v in H.items():
        c=row_cos(zx(v),Tv).mean(); res.append((p,k,c))
    # sigma check: sigma vs xs std
    R=H['sum']; print(' disp ratio', float((D.loc[idx]/cs(R).std(axis=1,ddof=0)).median()), float((D.loc[idx]/cs(R).std(axis=1,ddof=1)).median()))
df=pd.DataFrame(res,columns=['p','h','cos']).pivot(index='h',columns='p',values='cos'); print(df.round(4))
