import sys; sys.path.insert(0, r"F:\#Project\AlphaNova_S1C4")
import pandas as pd
from research.agents import peer_factor as pf
from lab import honest as h
def perp8(fn):
    def g(F):
        o = h.occupied_raw(F)
        return h.residualize(h.unit(fn(F)), [o["S08"]])
    return g
V = {"K8_W1440": lambda F: pf.pca_idio_rev(F, W=1440, K=8),
     "K10_W720": lambda F: pf.pca_idio_rev(F, W=720, K=10),
     "K8_W720_perp8": perp8(lambda F: pf.pca_idio_rev(F, W=720, K=8)),
     "K8_W1000": lambda F: pf.pca_idio_rev(F, W=1000, K=8)}
pf.CANDS.update(V)
print(pf.geometry(list(V)).round(3))
