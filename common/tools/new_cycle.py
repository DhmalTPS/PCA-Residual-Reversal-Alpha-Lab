"""Create a new cycle folder (never touches existing cycles) and optionally install its package.

  python common/tools/new_cycle.py S1/C5 --zip "C:/Users/me/Downloads/files.zip" --zip "C:/Users/me/Downloads/city_coords.zip"
  python common/tools/new_cycle.py S2/C1 --zip ... --activate

What it does:
  1. creates seasons/<S>/<C>/ (and, for a new season, seasons/<S>/SEASON.md, pot/ and an empty
     occupied_directions.py) with: vendor/ downloads/ incoming/ config/ strategies/ research/ experiments/
     reports/ submissions/{uploaded,not_novel,ready_not_uploaded}/ and CYCLE.md, STATUS.md, config/settings.json;
     seasons/<S>/SEASON.md and seasons/<S>/pot/ if the season is new;
  2. COPIES each zip into downloads/ (your originals stay where they are);
  3. safety-checks each zip (no absolute paths / '..'), extracts it into incoming/<zip name>/;
  4. if a zip contains the vendor package (runner.py + predictor.py), MOVES its contents into vendor/;
     if it contains only signal_cities_ambient_coords.parquet, merges it into vendor/signal_db/ (keeps the
     shipped and downloaded copies side by side);
  5. --activate switches ACTIVE_CYCLE to the new cycle.
It refuses to run if the cycle folder already exists, unless --install is given AND its vendor/ is still empty
(e.g. `python common/tools/new_cycle.py S1/C5 --install --zip ... --activate`). Nothing is ever overwritten.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CYCLE_MD = """# {cid} - cycle notes

* Opened: {today}
* Package: see downloads/ (originals) and vendor/ (as shipped - never edit vendor files)
* Status / decision log: STATUS.md (append newest first)
* Outcome: (fill at close: submissions, server verdicts, board rank, lessons) and add one line to <ROOT>/JOURNAL.md

## First steps (proven workflow, see <ROOT>/README.md)
1. python common/tools/set_cycle.py {cid}
2. python common/research_templates/r00_data_audit.py ; r01_target_identification.py --write
3. honest per-period screens (r06 / r09 / r16 templates), lab.honest.evaluate for every candidate
4. gates: precheck -> dynamic_check -> shuffle_gate -> VAULT (always, before any server check) -> freeze promote
"""
OCCUPIED_TEMPLATE = '''"""Season {sid} - directions known to be OCCUPIED (fill in as server novelty checks come back).

Same interface as seasons/S1/occupied_directions.py: occupied_raw(F) -> {{name: unit score}}, LEG_HL, SERVER_ANGLES.
"""
LEG_HL = {{}}
SERVER_ANGLES = {{}}


def occupied_raw(F):
    return {{}}
'''

SEASON_MD = """# Season {sid}

* Target / rules for this season: see the first cycle's vendor/COMPETITION.md
* pot/ holds frozen copies of OUR accepted submissions this season (they block similar ideas in later cycles)
* Cycles: see the C*/ folders; one-line outcomes in <ROOT>/JOURNAL.md
"""


def safe_extract(zp: Path, dest: Path) -> list[str]:
    with zipfile.ZipFile(zp) as z:
        names = z.namelist()
        bad = [n for n in names if n.startswith(("/", "\\")) or ".." in Path(n).parts or ":" in n]
        if bad:
            raise SystemExit(f"unsafe paths in {zp.name}: {bad[:5]}")
        z.extractall(dest)
    return names


def install(cyc: Path, extracted: Path) -> str:
    vendor = cyc / "vendor"
    roots = [p for p in extracted.rglob("runner.py") if (p.parent / "predictor.py").exists()]
    if roots:
        pkg = roots[0].parent
        for item in pkg.iterdir():
            dst = vendor / item.name
            if dst.exists():
                raise SystemExit(f"refusing to overwrite {dst}")
            shutil.move(str(item), str(dst))
        return f"vendor package from {pkg.name} -> vendor/"
    cities = list(extracted.rglob("signal_cities_ambient_coords.parquet"))
    if cities:
        sdb = vendor / "signal_db"
        sdb.mkdir(parents=True, exist_ok=True)
        tag = f"signal_cities_ambient_coords.downloaded_{date.today().isoformat()}.parquet"
        shutil.copy2(cities[0], sdb / tag)
        main = sdb / "signal_cities_ambient_coords.parquet"
        if main.exists():
            import pandas as pd
            shipped = sdb / "signal_cities_ambient_coords.shipped.parquet"
            if not shipped.exists():
                shutil.copy2(main, shipped)
            a, b = pd.read_parquet(shipped), pd.read_parquet(cities[0])
            pd.concat([b, a[~a["predictor_id"].isin(b["predictor_id"])]], ignore_index=True).to_parquet(main)
            return f"city map merged into vendor/signal_db/ ({tag} kept)"
        shutil.copy2(cities[0], main)
        return "city map -> vendor/signal_db/"
    return "unrecognised archive: left in incoming/"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cycle", help="e.g. S1/C5")
    ap.add_argument("--zip", action="append", default=[])
    ap.add_argument("--activate", action="store_true")
    ap.add_argument("--install", action="store_true",
                    help="install --zip packages into an EXISTING cycle whose vendor/ is still empty (e.g. a pre-created C5)")
    a = ap.parse_args(argv)
    cid = a.cycle.strip().strip("/")
    sid, cname = cid.split("/", 1)
    season = ROOT / "seasons" / sid
    cyc = season / cname
    if cyc.exists() and a.install:
        if any((cyc / "vendor").iterdir()):
            print(f"{cyc / 'vendor'} is not empty - refusing (nothing is ever overwritten).")
            return 1
        for z in a.zip:
            zp = Path(z)
            dl = cyc / "downloads" / zp.name
            if not dl.exists():
                shutil.copy2(zp, dl)
            dest = cyc / "incoming" / zp.stem
            safe_extract(zp, dest)
            print(f"{zp.name}: {install(cyc, dest)}")
        if a.activate:
            (ROOT / "ACTIVE_CYCLE").write_text(cid + "\n", encoding="utf-8")
            print(f"active cycle -> {cid}")
        return 0
    if cyc.exists():
        print(f"{cyc} already exists - refusing (nothing is ever overwritten). To install a package into it use --install.")
        return 1
    for d in ("vendor", "downloads", "incoming", "config", "strategies", "research", "experiments", "reports",
              "submissions/uploaded", "submissions/not_novel", "submissions/ready_not_uploaded"):
        (cyc / d).mkdir(parents=True, exist_ok=True)
    (season / "pot").mkdir(parents=True, exist_ok=True)
    if not (season / "SEASON.md").exists():
        (season / "SEASON.md").write_text(SEASON_MD.format(sid=sid), encoding="utf-8")
    if not (season / "occupied_directions.py").exists():
        (season / "occupied_directions.py").write_text(OCCUPIED_TEMPLATE.format(sid=sid), encoding="utf-8")
    (cyc / "CYCLE.md").write_text(CYCLE_MD.format(cid=cid, today=date.today().isoformat()), encoding="utf-8")
    (cyc / "STATUS.md").write_text(f"# STATUS - {cid}\n\n(newest first)\n", encoding="utf-8")
    (cyc / "config" / "settings.json").write_text('{\n  "_doc": "overrides of common/config/defaults.json for ' + cid + '"\n}\n', encoding="utf-8")
    for z in a.zip:
        zp = Path(z)
        shutil.copy2(zp, cyc / "downloads" / zp.name)
        dest = cyc / "incoming" / zp.stem
        safe_extract(zp, dest)
        print(f"{zp.name}: {install(cyc, dest)}")
    if a.activate:
        (ROOT / "ACTIVE_CYCLE").write_text(cid + "\n", encoding="utf-8")
        print(f"active cycle -> {cid}")
    print(f"created {cyc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
