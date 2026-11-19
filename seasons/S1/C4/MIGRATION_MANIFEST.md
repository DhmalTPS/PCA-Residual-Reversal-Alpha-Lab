# Migration manifest - restructure to common/ + seasons/<S>/<C>/ layout

Date (UTC): 2026-10-08T06:07:49+00:00. Only MOVES were performed; no file was deleted.

| old path | new path | action |
|---|---|---|
| `lab` | `common\lab` | moved |
| `tools` | `common\tools` | moved |
| `tests` | `common\tests` | moved |
| `data_synth` | `common\synthetic` | moved |
| `pot` | `seasons\S1\pot` | moved |
| `runner.py` | `seasons\S1\C4\vendor\runner.py` | moved |
| `predictor.py` | `seasons\S1\C4\vendor\predictor.py` | moved |
| `walkforward.py` | `seasons\S1\C4\vendor\walkforward.py` | moved |
| `evaluation.py` | `seasons\S1\C4\vendor\evaluation.py` | moved |
| `data_loader.py` | `seasons\S1\C4\vendor\data_loader.py` | moved |
| `city_tools.py` | `seasons\S1\C4\vendor\city_tools.py` | moved |
| `demo.py` | `seasons\S1\C4\vendor\demo.py` | moved |
| `demo_engineered.py` | `seasons\S1\C4\vendor\demo_engineered.py` | moved |
| `demo.ipynb` | `seasons\S1\C4\vendor\demo.ipynb` | moved |
| `README.md` | `seasons\S1\C4\vendor\README.md` | moved |
| `COMPETITION.md` | `seasons\S1\C4\vendor\COMPETITION.md` | moved |
| `METRICS.md` | `seasons\S1\C4\vendor\METRICS.md` | moved |
| `requirements.txt` | `seasons\S1\C4\vendor\requirements.txt` | moved |
| `data` | `seasons\S1\C4\vendor\data` | moved |
| `signal_db` | `seasons\S1\C4\vendor\signal_db` | moved |
| `__pycache__` | `seasons\S1\C4\vendor\__pycache__` | moved |
| `downloads` | `seasons\S1\C4\downloads` | moved |
| `_incoming` | `seasons\S1\C4\incoming` | moved |
| `config` | `seasons\S1\C4\config` | moved |
| `strategies` | `seasons\S1\C4\strategies` | moved |
| `research` | `seasons\S1\C4\research` | moved |
| `experiments` | `seasons\S1\C4\experiments` | moved |
| `reports` | `seasons\S1\C4\reports` | moved |
| `submit` | `seasons\S1\C4\submissions` | moved |
| `STATUS.md` | `seasons\S1\C4\STATUS.md` | moved |
| `PROJECT.md` | `seasons\S1\C4\PROJECT_NOTES.md` | moved |
| `submit/s09_pca_idio_reversal.py` | `seasons/S1/C4/submissions/uploaded/s09_pca_idio_reversal.py` | moved |
| `submit/s09_pca_idio_reversal.checklist.md` | `seasons/S1/C4/submissions/uploaded/s09_pca_idio_reversal.checklist.md` | moved |
| `submit/s09_pca_idio_reversal_perp_s08True.py` | `seasons/S1/C4/submissions/ready_not_uploaded/s09_pca_idio_reversal_perp_s08True.py` | moved |
| `submit/s09_pca_idio_reversal_perp_s08True.checklist.md` | `seasons/S1/C4/submissions/ready_not_uploaded/s09_pca_idio_reversal_perp_s08True.checklist.md` | moved |
| `submit/discarded/s06_reversal_composite.checklist.md` | `seasons/S1/C4/submissions/not_novel/s06_reversal_composite.checklist.md` | moved |
| `submit/discarded/s06_reversal_composite_notnovel.py` | `seasons/S1/C4/submissions/not_novel/s06_reversal_composite_notnovel.py` | moved |
| `submit/discarded/s07_extremeness_v1.checklist.md` | `seasons/S1/C4/submissions/not_novel/s07_extremeness_v1.checklist.md` | moved |
| `submit/discarded/s07_extremeness_v1_notnovel.py` | `seasons/S1/C4/submissions/not_novel/s07_extremeness_v1_notnovel.py` | moved |
| `submit/discarded/s07_extremeness_v2.checklist.md` | `seasons/S1/C4/submissions/not_novel/s07_extremeness_v2.checklist.md` | moved |
| `submit/discarded/s07_extremeness_v2_notnovel.py` | `seasons/S1/C4/submissions/not_novel/s07_extremeness_v2_notnovel.py` | moved |
| `submit/discarded/s08_two_leg_reversal.checklist.md` | `seasons/S1/C4/submissions/not_novel/s08_two_leg_reversal.checklist.md` | moved |
| `submit/discarded/s08_two_leg_reversal_notnovel.py` | `seasons/S1/C4/submissions/not_novel/s08_two_leg_reversal_notnovel.py` | moved |
| `submit/discarded/` | `(empty folder removed after all files moved to not_novel/)` | emptied |