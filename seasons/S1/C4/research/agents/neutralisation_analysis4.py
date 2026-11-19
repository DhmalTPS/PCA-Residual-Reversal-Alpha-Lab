"""LOOK-AHEAD ANALYSIS: vintage hypothesis - is the old-row gap T(p) vs zx(fwd X(p)) explained by X/T revisions across files?"""
import sys; sys.path.insert(0,'.')
import pandas as pd, numpy as np
from lab.paths import data_root
from lab.labels import window_sum
from lab.metrics import row_cos
from lab.honest import zx
r=data_root()
L=lambda p,k: pd.read_parquet(r/f'{p}.small.{k}.parquet')
for a,b in [('006','007'),('007','008'),('009','010'),('021','022'),('024','025')]:
    Xa,Xb,Ta,Tb=L(a,'returns'),L(b,'returns'),L(a,'target'),L(b,'target')
    print(a,b,'index start',Xa.index[0],Xb.index[0],'end',Xa.index[-1],Xb.index[-1], 'cols eq', list(Xa.columns)==list(Xb.columns))
    ix=Xa.index.intersection(Xb.index)[-6000:-200]
    print('  X same?', float(np.nanmax(np.abs(Xa.loc[ix].values-Xb.loc[ix].values))), ' T cos', float(row_cos(Ta.loc[ix],Tb.loc[ix]).mean()))
    Zb=zx(window_sum(Xb.loc[ix[0]:],120,1)).loc[ix]; Za=zx(window_sum(Xa.loc[ix[0]:],120,1)).loc[ix]
    print('  cos(Tb, zXb)',float(row_cos(Tb.loc[ix],Zb).mean()),' cos(Tb,zXa)',float(row_cos(Tb.loc[ix],Za).mean()),' cos(Ta,zXa)',float(row_cos(Ta.loc[ix],Za).mean()))
