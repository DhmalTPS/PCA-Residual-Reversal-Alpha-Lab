import sys, json; sys.path.insert(0, r"F:\#Project\AlphaNova_S1C4")
import numpy as np
from research.agents import peer_factor as pf
from lab import honest as h
def perp8(fn):
    def g(F):
        o = h.occupied_raw(F)
        return h.residualize(h.unit(fn(F)), [o["S08"]])
    return g
V = {"K5_perp8": perp8(lambda F: pf.pca_idio_rev(F, K=5)),
     "K8_W720": lambda F: pf.pca_idio_rev(F, W=720, K=8),
     "K12_W336": lambda F: pf.pca_idio_rev(F, W=336, K=12),
     "K5_L240": lambda F: pf.pca_idio_rev(F, K=5, L=240),
     "K8_W336": lambda F: pf.pca_idio_rev(F, W=336, K=8)}
V.update({"K10_W720": lambda F: pf.pca_idio_rev(F, W=720, K=10),
          "K8_W1000": lambda F: pf.pca_idio_rev(F, W=1000, K=8),
          "K8_W720_perp8": perp8(lambda F: pf.pca_idio_rev(F, W=720, K=8))})
for n in sys.argv[1:]:
    m = h.evaluate(V[n], f"peer_pca_{n}", hl=672.0, warm=int(4 * 672 + 1200), notes="PEER_FACTOR agent")
    m["admissible"] = bool(h.admissible(m)); print("ADMISSIBLE", n, m["admissible"])
    with open(r"F:\#Project\AlphaNova_S1C4\research\agents\peer_factor_results.jsonl", "a") as f:
        f.write(json.dumps({k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in m.items()}) + "\n")
