"""
Hardware-validation kit, step 3: score measured success vs frozen predictions.
usage: python3 hw_validation/score.py MANIFEST COUNTS
Success = every pair's two bits equal in the same shot. Reports MAE, bias, rank correlation
for v4.1 and GNN, split single vs multi-bell, plus a per-circuit table.
"""
import json, sys
import numpy as np

man = json.load(open(sys.argv[1])); cnt = json.load(open(sys.argv[2]))["results"]


def success(counts, n):
    tot = sum(counts.values()); ok = 0
    for bstr, c in counts.items():
        s = bstr.replace(" ", "")[::-1]            # s[k] = clbit k
        if all(s[2 * i] == s[2 * i + 1] for i in range(n)):
            ok += c
    return ok / tot


rows = []
for c in man["circuits"]:
    if c["id"] in cnt:
        rows.append((c, success(cnt[c["id"]], len(c["pairs"]))))


def rank(x): return np.argsort(np.argsort(x)).astype(float)


def report(name, sel):
    if len(sel) < 3: return
    y = np.array([m for _, m in sel])
    print(f"\n{name} (n={len(sel)}): measured mean {y.mean():.3f}")
    for k in ("v41", "gnn"):
        p = np.array([c[k] for c, _ in sel])
        rho = np.corrcoef(rank(p), rank(y))[0, 1]
        print(f"  {k:4s}: MAE {np.abs(p-y).mean()*100:5.2f} pts | bias {(p-y).mean()*100:+5.2f} pts | rank corr {rho:.2f}")
    v, g = [np.abs(np.array([c[k] for c, _ in sel]) - y).mean() for k in ("v41", "gnn")]
    print(f"  GNN vs v4.1 MAE: {(1-g/v)*100:+.1f}% {'(better)' if g < v else '(worse)'}")


report("ALL", rows)
report("single bell", [r for r in rows if r[0]["kind"] == "bell"])
report("multi-bell", [r for r in rows if r[0]["kind"] == "multibell"])
print("\nid           hops      measured   v4.1   gnn")
for c, m in rows:
    print(f"{c['id']:12s} {str(c['hops']):9s} {m:8.3f} {c['v41']:6.3f} {c['gnn']:6.3f}")
