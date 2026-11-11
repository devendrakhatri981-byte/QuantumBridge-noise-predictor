"""Sequential (no multiprocessing) grower for entry102; time-budgeted. usage: N CHIP"""
import json, os, random, sys, time
import entry102_grow_mixed as m, emulator_v3_routing as em
n, chip = int(sys.argv[1]), sys.argv[2]
budget = float(os.environ.get("GROW_BUDGET", "100"))
P = f"quantumbridge_data/entry102_mixed_{chip}.json"
recs = json.load(open(P)) if os.path.exists(P) else []
mine = [r for r in recs if r["chip"] == chip]
rng = random.Random(102 + hash(chip) % 10000 + len(mine) + int(time.time()) % 1000)
sigs = {(r["chip"], tuple(sorted(tuple(c) for c in r["components"]))) for r in recs}
gr = em.build_connectivity_graph(em.load_calibration(chip), chip)
t0 = time.time(); added = 0
while added < n and time.time() - t0 < budget:
    t = m.sample_task(chip, 127, gr, rng)
    if t is None: continue
    sig = (chip, tuple(sorted(tuple(c) for c in t[1])))
    if sig in sigs: continue
    sigs.add(sig); recs.append(m.run_mixed(t)); added += 1
    if added % 5 == 0: m.g._atomic_save(recs, P)
m.g._atomic_save(recs, P)
print("added", added, "in", round(time.time() - t0), "s", flush=True)
