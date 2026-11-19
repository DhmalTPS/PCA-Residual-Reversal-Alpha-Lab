"""STATIC compliance gate for a strategy / submission file (mirrors the server's entry checks).

  1. PEEK_THE_FUTURE scan: line-by-line, comments INCLUDED (the server scans them too).
     Our pattern list is deliberately broader than the published examples.
  2. CODE_OUTSIDE_CLASS: AST check of every module-level statement against the allow-list.
  3. Hygiene: no import of the research library, no reference to target/dispersion/returns
     files or the signal database, exactly one Predictor subclass with train() and predict().

Exit code 0 = PASS, 1 = FAIL. Warnings never fail.
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

FAIL_PATTERNS = [
    (r"\.shift\(\s*-", "negative shift"),
    (r"\bshift\(\s*periods\s*=\s*-", "negative shift"),
    (r"periods\s*=\s*-\s*\d", "negative periods"),
    (r"\.diff\(\s*-", "negative diff"),
    (r"pct_change\(\s*-", "negative pct_change"),
    (r"center\s*=\s*True", "centered window"),
    (r"\bbfill\b|\bbackfill\b", "backward fill"),
    (r"method\s*=\s*['\"](bfill|backfill|nearest)['\"]", "fill method uses later rows"),
    (r"\bfiltfilt\b|\bsosfiltfilt\b", "zero-phase filter"),
    (r"\bnp\.roll\b|\.roll\(", "wrap-around roll"),
    (r"\[\s*::\s*-\s*1\s*\]", "reversed slice"),
    (r"\bflip(ud|lr)?\(", "array flip"),
    (r"\bfft\b|\brfft\b|\bifft\b", "spectral (non-causal) transform"),
    (r"\.interpolate\(", "interpolation uses later rows"),
    (r"\bsavgol", "Savitzky-Golay (centered)"),
    (r"\bgaussian_filter|\buniform_filter|\bmedian_filter|\bconvolve\(.*mode\s*=\s*['\"]same", "centered kernel"),
    (r"\.iloc\[[^\]]*\+\s*\d", "positive positional offset"),
    (r"\.lead\(", "lead operator"),
    (r"\breversed\(", "reversed iteration (review)"),
]
WARN_PATTERNS = [
    (r"\bfuture\b|\bforward\b|look-?ahead|\bpeek", "suspicious wording (the scan reads comments)"),
    (r"\bcumsum\b|\bcumprod\b|\bexpanding\(|\bewm\(", "infinite-memory op: check ROW_MASK (prefer finite windows)"),
    (r"\bcummax\b|\bcummin\b", "infinite-memory op"),
    (r"np\.random\.(rand|randn|random|normal|uniform|choice)\(", "unseeded global RNG"),
    (r"\bdatetime\.now\b|\btime\.time\b", "wall-clock dependence"),
]
FORBIDDEN_REFS = [
    (r"\bfrom\s+lab\b|\bimport\s+lab\b", "imports the research library"),
    (r"target\.parquet|dispersion\.parquet|returns\.parquet|signal_db|signal_cities|\.npy\b", "reads data the predictor must not see"),
    (r"read_parquet|read_csv|open\(", "file I/O inside a submission"),
]

CONFIG_CALLS = {
    "warnings.filterwarnings", "warnings.simplefilter", "filterwarnings", "simplefilter",
    "logging.basicConfig", "logging.getLogger", "basicConfig", "getLogger",
    "pd.set_option", "pandas.set_option", "set_option", "np.seterr", "numpy.seterr", "seterr",
}


def _dotted(node) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_dotted(node.value)}.{node.attr}"
    if isinstance(node, ast.Call):
        return _dotted(node.func)
    return "?"


def _calls(node) -> list[ast.Call]:
    return [n for n in ast.walk(node) if isinstance(n, ast.Call)]


def _is_config_call(call: ast.Call, depth: int = 0) -> bool:
    name = _dotted(call.func)
    if name in CONFIG_CALLS or name.split(".")[-1] in {"filterwarnings", "simplefilter", "basicConfig", "set_option", "seterr"} and name.split(".")[0] in {"warnings", "logging", "pd", "pandas", "np", "numpy"}:
        return True
    # up to two chained method calls on the result of one config call: getLogger(...).setLevel(...)
    if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Call) and depth < 2:
        return _is_config_call(call.func.value, depth + 1)
    return False


def _predictor_classes(tree) -> list[str]:
    out = []
    for n in tree.body:
        if isinstance(n, ast.ClassDef) and any(_dotted(b).split(".")[-1] == "Predictor" or _dotted(b) in out for b in n.bases):
            out.append(n.name)
    return out


def ast_check(tree) -> tuple[list[str], list[str]]:
    fails, warns = [], []
    pclasses = set(_predictor_classes(tree))
    if len(pclasses) != 1:
        fails.append(f"expected exactly one Predictor subclass, found {sorted(pclasses)}")
    for i, n in enumerate(tree.body):
        where = f"line {n.lineno}"
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            continue
        if i == 0 and isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str):
            continue
        if isinstance(n, ast.ClassDef):
            if n.name not in pclasses:
                fails.append(f"{where}: non-Predictor class '{n.name}' at module level")
            else:
                meths = {b.name for b in n.body if isinstance(b, ast.FunctionDef)}
                for req in ("train", "predict"):
                    if req not in meths:
                        fails.append(f"{where}: {n.name} does not define {req}()")
            continue
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fails.append(f"{where}: top-level def '{n.name}'")
            continue
        if isinstance(n, ast.If):
            t = n.test
            ok = (isinstance(t, ast.Compare) and len(t.ops) == 1 and isinstance(t.ops[0], ast.Eq)
                  and {_dotted(t.left), _dotted(t.comparators[0]) if not isinstance(t.comparators[0], ast.Constant) else "__main__"} >= {"__name__"}
                  and any(isinstance(x, ast.Constant) and x.value == "__main__" for x in (t.left, t.comparators[0])))
            if not ok:
                fails.append(f"{where}: module-level if (only `if __name__ == \"__main__\":` is allowed)")
            elif n.orelse:
                fails.append(f"{where}: `if __name__ == \"__main__\":` must not have an else branch")
            continue
        if isinstance(n, ast.Try):
            handlers_ok = all(h.type is not None and "ImportError" in ast.unparse(h.type) for h in n.handlers)
            body_ok = all(isinstance(b, (ast.Import, ast.ImportFrom)) for b in n.body)
            if not (handlers_ok and body_ok):
                fails.append(f"{where}: try/except at module level (only an ImportError import guard is allowed)")
            continue
        calls = _calls(n)
        if isinstance(n, ast.Assign) and any(isinstance(v, ast.Lambda) for v in ast.walk(n.value)):
            fails.append(f"{where}: lambda bound at module level (a helper function in disguise)")
            continue
        if not calls:
            warns.append(f"{where}: call-free module-level statement (warns on server): {ast.unparse(n)[:60]}")
            continue
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call) and _dotted(n.value.func) in pclasses:
            inner = [c for c in calls if c is not n.value]
            if all(_is_config_call(c) for c in inner):
                continue
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and _is_config_call(n.value):
            continue
        fails.append(f"{where}: module-level statement makes a call: {ast.unparse(n)[:80]}")
    return fails, warns


def scan(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    fails, warns = [], []
    for ln, line in enumerate(lines, 1):
        stripped = line.lstrip()
        if stripped.startswith("!") or stripped.startswith("%"):
            continue
        for pat, why in FAIL_PATTERNS:
            if re.search(pat, line):
                fails.append(f"line {ln}: PEEK_THE_FUTURE? {why}: {line.strip()[:90]}")
        for pat, why in FORBIDDEN_REFS:
            if re.search(pat, line):
                fails.append(f"line {ln}: {why}: {line.strip()[:90]}")
        for pat, why in WARN_PATTERNS:
            if re.search(pat, line, flags=re.IGNORECASE):
                warns.append(f"line {ln}: {why}: {line.strip()[:90]}")
    try:
        tree = ast.parse(text)
        f2, w2 = ast_check(tree)
        fails += [f"CODE_OUTSIDE_CLASS: {x}" if "Predictor subclass" not in x and "does not define" not in x else x for x in f2]
        warns += w2
    except SyntaxError as e:
        fails.append(f"syntax error: {e}")
    return {"file": str(path), "fails": fails, "warns": warns, "passed": not fails}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    a = ap.parse_args(argv)
    rc = 0
    for f in a.files:
        r = scan(Path(f))
        status = "PASS" if r["passed"] else "FAIL"
        print(f"[{status}] {f}")
        for x in r["fails"]:
            print(f"   FAIL  {x}")
        for x in r["warns"]:
            print(f"   warn  {x}")
        rc |= 0 if r["passed"] else 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
