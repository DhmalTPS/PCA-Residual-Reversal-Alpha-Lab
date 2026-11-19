"""LOOK-AHEAD ANALYSIS: where does T deviate from zx(fwd 120h sum, offset 1)? All dev files, by segment."""
import sys; sys.path.insert(0,'.')
import pandas as pd, numpy as np
from lab.paths import data_root
from lab.labels import window_sum
from lab.metrics import row_cos
from lab.honest import zx, dev_periods
r=data_root(); out=[]
for p in dev_periods():
    X=pd.read_parquet(r/f'{p}.small.returns.parquet'); T=pd.read_parquet(r/f'{p}.small.target.parquet')
    n=len(X); X=X.iloc[-6000:]; T=T.loc[X.index]
    Z=zx(window_sum(X,120,1)); c=row_cos(Z,T)
    nn=X.notna().sum(axis=1)
    seg=lambda a,b: float(c.iloc[a:b].mean())
    bad=c.iloc[:-130]; 
    out.append(dict(p=p,n=n,old=seg(0,3000),pre=seg(3000,5070),val_noend=seg(5200,5870),last130=seg(5870,6000),
        frac_bad=float((bad<0.999).mean()),nan_tickers_val=float((nn.iloc[-800:]<20).mean()), Tzero=float((T.iloc[-800:].std(axis=1)<1e-8).mean())))
print(pd.DataFrame(out).round(4).to_string())
