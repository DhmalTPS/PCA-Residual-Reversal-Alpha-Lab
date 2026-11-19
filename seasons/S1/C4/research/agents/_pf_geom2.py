import sys; sys.path.insert(0, r"F:\#Project\AlphaNova_S1C4")
import pandas as pd
from research.agents import peer_factor as pf
from lab import honest as h
def perp(fn):
    def g(F):
        o = h.occupied_raw(F)
        return h.residualize(h.unit(fn(F)), [o["S06"], o["S07"], o["S08"]])
    return g
def perp8(fn):
    def g(F):
        o = h.occupied_raw(F)
        return h.residualize(h.unit(fn(F)), [o["S08"]])
    return g
V = {}
for K in (5, 8, 12):
    for W in (336, 720, 1440):
        V[f"K{K}_W{W}"] = (lambda K, W: lambda F: pf.pca_idio_rev(F, W=W, K=K))(K, W)
V["K5_L240"] = lambda F: pf.pca_idio_rev(F, K=5, L=240)
V["K8_L240"] = lambda F: pf.pca_idio_rev(F, K=8, L=240)
V["K5_L48"] = lambda F: pf.pca_idio_rev(F, K=5, L=48)
V["K8_m4"] = lambda F: pf.pca_idio_rev(F, K=8, kind="m4")
V["K5_perp"] = perp(lambda F: pf.pca_idio_rev(F, K=5))
V["K5_perp8"] = perp8(lambda F: pf.pca_idio_rev(F, K=5))
pf.CANDS.update(V)
pd.set_option("display.width", 200)
print(pf.geometry(list(V)).round(3))
