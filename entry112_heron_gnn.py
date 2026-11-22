"""
Entry 112: GNN trained ONLY on simulated Heron (fez/marrakesh/kingston snapshots, clean routes) and tested on
(A) leave-one-snapshot-out in simulation and (B) the 20 REAL ibm_fez single-pair circuits (frozen manifest, live calibration,
own-stats normalization from unlabeled live pairs). Hyperparameters fixed a priori (same as earlier GNNs); nothing tuned on hardware.
usage: python3 entry112_heron_gnn.py build | A | B   (NSEEDS env, default 5)
"""
import json, os, random, sys, time
from functools import partial
import numpy as np
import jax, jax.numpy as jnp, optax

MAX_N, MAX_E, NNF, NEF, H, EPOCHS = 30, 30, 6, 4, 16, 150
CHIPS = ("fez", "marrakesh", "kingston"); GRAPHS = "quantumbridge_data/entry112_graphs.json"
NSEEDS = int(os.environ.get("NSEEDS", "5"))


def build():
    sys.path.insert(0, "."); import api.main as M, entry089_grow_multibell as g89, emulator_v3_routing as em
    M.BACKENDS.update({k: v for k, v in g89.BACKENDS.items() if k not in M.BACKENDS}); out = []
    for chip in CHIPS:
        b, gr, coh = M.get_backend_ctx(chip)
        for r in json.load(open(f"quantumbridge_data/entry111_heron_{chip}.json")):
            g = M.graph_for_record(b, gr, coh, r); g.update({"chip": chip, "set": "sim", "hop": r["bfs_hop_distance"],
                                                              "v41": r["v4_1_prediction"], "y": r["aer_ground_truth"]}); out.append(g)
    man = json.load(open("hw_validation/manifest_fez_live.json")); cnt = {}
    for f in ("counts_ibm_fez_fez_live.json", "counts_ibm_fez_fez_live_s5.json"): cnt.update(json.load(open("hw_validation/" + f))["results"])
    b, gr, coh = M.get_backend_ctx("fez_live"); rng = random.Random(3); nq = b.num_qubits
    for c in man["circuits"]:
        if c["kind"] != "bell": continue
        a, bb = c["pairs"][0]; rec = {"chip": "fez_live", "kind": "bell", "pairs_flat": [a, bb], "logical_pairs": [[a, bb]], "bfs_hop_distance": c["hops"][0]}
        g = M.graph_for_record(b, gr, coh, rec); cn = cnt[c["id"]]; tot = sum(cn.values())
        y = sum(v for k, v in cn.items() if k.replace(" ", "")[::-1][0] == k.replace(" ", "")[::-1][1]) / tot
        g.update({"chip": "fez_live", "set": "hw", "hop": c["hops"][0], "v41": c["v41"], "y": y, "id": c["id"], "eagle_gnn": c["gnn"]}); out.append(g)
    dead_q = set(json.load(open("quantumbridge_data/coherence_fez_live.json")).get("missing_qubits_filled_with_median", []))
    n = 0
    while n < 100:                                   # unlabeled live pairs for own-stats
        a, bb = rng.sample(range(nq), 2); p = em.shortest_path(gr, a, bb)
        if p is None or len(p) - 1 > 8 or a in dead_q or bb in dead_q: continue
        rec = {"chip": "fez_live", "kind": "bell", "pairs_flat": [a, bb], "logical_pairs": [[a, bb]], "bfs_hop_distance": len(p) - 1}
        g = M.graph_for_record(b, gr, coh, rec); g.update({"chip": "fez_live", "set": "stats", "hop": len(p) - 1, "v41": 0.9, "y": 0.0}); out.append(g); n += 1
    json.dump(out, open(GRAPHS, "w")); print("graphs:", {s: sum(1 for x in out if x["set"] == s) for s in ("sim", "hw", "stats")})


def pad(g):
    nf = np.zeros((MAX_N, NNF), np.float32); nm = np.zeros(MAX_N, np.float32); n = g["n_nodes"]
    nf[:n] = np.array(g["nodes"]); nm[:n] = 1
    src = np.zeros(MAX_E, np.int32); dst = np.zeros(MAX_E, np.int32); ef = np.zeros((MAX_E, NEF), np.float32); em_ = np.zeros(MAX_E, np.float32)
    for i, e in enumerate(g["edges"][:MAX_E]): src[i], dst[i] = e[0], e[1]; ef[i] = e[2:]; em_[i] = 1
    return nf, nm, src, dst, ef, em_, np.array([g["hop"], g["v41"]], np.float32)


def init(key):
    ks = jax.random.split(key, 7)
    lin = lambda k, i, o: {"W": jax.random.normal(k, (i, o)) * jnp.sqrt(2.0 / i), "b": jnp.zeros(o)}
    return {"ne": lin(ks[0], NNF, H), "m1": lin(ks[1], 2 * H + NEF, H), "u1": lin(ks[2], 2 * H, H), "m2": lin(ks[3], 2 * H + NEF, H),
            "u2": lin(ks[4], 2 * H, H), "r1": lin(ks[5], H + 2, H), "r2": lin(ks[6], H, 1)}


L = lambda p, x: x @ p["W"] + p["b"]


def mp(pm, pu, h, nm, src, dst, ef, em_):
    msg = jax.nn.relu(L(pm, jnp.concatenate([h[src], h[dst], ef], -1))) * em_[:, None]
    agg = jnp.zeros((MAX_N, H)).at[dst].add(msg) / (jnp.zeros(MAX_N).at[dst].add(em_) + 1e-6)[:, None]
    return jax.nn.relu(L(pu, jnp.concatenate([h, agg], -1))) * nm[:, None]


def fwd(p, nf, nm, src, dst, ef, em_, gl):
    h = jax.nn.relu(L(p["ne"], nf)) * nm[:, None]; h = mp(p["m1"], p["u1"], h, nm, src, dst, ef, em_); h = mp(p["m2"], p["u2"], h, nm, src, dst, ef, em_)
    pooled = (h * nm[:, None]).sum(0) / (nm.sum() + 1e-6)
    return jax.nn.sigmoid(L(p["r2"], jax.nn.relu(L(p["r1"], jnp.concatenate([pooled, gl]))))[0])


fb = jax.vmap(fwd, in_axes=(None, 0, 0, 0, 0, 0, 0, 0))
loss = lambda p, b: jnp.mean(jnp.abs(fb(p, *b[:7]) - b[7]))


@partial(jax.jit, static_argnums=(3,))
def step(p, s, b, opt):
    g = jax.grad(loss)(p, b); u, s = opt.update(g, s, p); return optax.apply_updates(p, u), s


def prep(gs):
    P = [pad(g) for g in gs]; st = lambda i: np.stack([x[i] for x in P])
    return [st(i) for i in range(7)] + [np.array([g["y"] for g in gs], np.float32)]


def norm(D, gs, stat_gs):
    """own-stats per chip from the given unlabeled graphs; returns normalized copy"""
    nf, ef = D[0].copy(), D[4].copy()
    for chip in {g["chip"] for g in gs}:
        idx = np.array([i for i, g in enumerate(gs) if g["chip"] == chip]); sg = [g for g in stat_gs if g["chip"] == chip]
        SD = prep(sg) if sg else None
        nmk = SD[1].astype(bool); emk = SD[5].astype(bool)
        mu, sd = SD[0][nmk].mean(0), SD[0][nmk].std(0) + 1e-6; emu, esd = SD[4][emk].mean(0), SD[4][emk].std(0) + 1e-6
        nf[idx] = (D[0][idx] - mu) / sd; ef[idx] = (D[4][idx] - emu) / esd
    return nf, ef


def train_predict(trD, teD, seed):
    gm, gs_ = trD[6].mean(0), trD[6].std(0) + 1e-6
    tr = [trD[0], trD[1], trD[2], trD[3], trD[4], trD[5], (trD[6] - gm) / gs_, trD[7]]
    te = [teD[0], teD[1], teD[2], teD[3], teD[4], teD[5], (teD[6] - gm) / gs_, teD[7]]
    opt = optax.adam(3e-3); p = init(jax.random.PRNGKey(seed)); s = opt.init(p); rng = np.random.RandomState(seed); n = len(tr[7])
    for _ in range(EPOCHS):
        perm = rng.permutation(n)
        for i in range(0, n, 32):
            j = perm[i:i + 32]; p, s = step(p, s, tuple(x[j] for x in tr), opt)
    return np.array(fb(p, *[jnp.array(x) for x in te[:7]]))


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "build": build(); sys.exit()
    G = json.load(open(GRAPHS)); sim = [g for g in G if g["set"] == "sim"]; hw = [g for g in G if g["set"] == "hw"]; st = [g for g in G if g["set"] == "stats"]
    t0 = time.time()
    if mode == "A":                                   # leave-one-snapshot-out in simulation (own-stats)
        for hold in CHIPS:
            tr = [g for g in sim if g["chip"] != hold]; te = [g for g in sim if g["chip"] == hold]
            allg = tr + te; D = prep(allg); nf, ef = norm(D, allg, allg); D[0], D[4] = nf, ef
            ntr = len(tr); TR = [x[:ntr] for x in D]; TE = [x[ntr:] for x in D]; y = TE[7]; v = TE[6][:, 1]; res = []
            for sd in range(NSEEDS):
                pr = train_predict(TR, TE, sd); res.append(np.abs(pr - y).mean() * 100)
            print(f"A hold {hold:10s} n={len(te)} v4.1 MAE {np.abs(v-y).mean()*100:5.2f} | GNN(Heron-trained) MAE {np.mean(res):5.2f} +- {np.std(res):.2f} (seeds {[round(r,2) for r in res]}) | {time.time()-t0:.0f}s", flush=True)
    else:                                             # B: real hardware
        allg = sim + hw + st; D = prep(allg)
        stat_for_norm = sim + st                      # hw graphs normalized with unlabeled live-pair stats (label-free)
        nf, ef = D[0].copy(), D[4].copy(); idx_sim = np.array([i for i, g in enumerate(allg) if g["set"] == "sim"])
        for chip in CHIPS:
            ix = np.array([i for i, g in enumerate(allg) if g["set"] == "sim" and g["chip"] == chip]); SD = prep([allg[i] for i in ix])
            nmk, emk = SD[1].astype(bool), SD[5].astype(bool)
            mu, sd_, emu, esd = SD[0][nmk].mean(0), SD[0][nmk].std(0) + 1e-6, SD[4][emk].mean(0), SD[4][emk].std(0) + 1e-6
            nf[ix] = (D[0][ix] - mu) / sd_; ef[ix] = (D[4][ix] - emu) / esd
        ix = np.array([i for i, g in enumerate(allg) if g["set"] in ("hw", "stats")]); SD = prep([g for g in allg if g["set"] == "stats"])
        nmk, emk = SD[1].astype(bool), SD[5].astype(bool)
        mu, sd_, emu, esd = SD[0][nmk].mean(0), SD[0][nmk].std(0) + 1e-6, SD[4][emk].mean(0), SD[4][emk].std(0) + 1e-6
        nf[ix] = (D[0][ix] - mu) / sd_; ef[ix] = (D[4][ix] - emu) / esd
        D[0], D[4] = nf, ef
        itr = np.array([i for i, g in enumerate(allg) if g["set"] == "sim"]); ite = np.array([i for i, g in enumerate(allg) if g["set"] == "hw"])
        TR = [x[itr] for x in D]; TE = [x[ite] for x in D]; y = TE[7]; v = TE[6][:, 1]; res = []; preds = []
        for sd in range(NSEEDS):
            pr = train_predict(TR, TE, sd); preds.append(pr); res.append(np.abs(pr - y).mean() * 100)
        pm = np.mean(preds, 0); eg = np.array([g["eagle_gnn"] for g in hw])
        print(f"B real ibm_fez singles n={len(y)} measured mean {y.mean():.3f} std {y.std()*100:.1f} pts")
        print(f"  constant median MAE {np.abs(y-np.median(y)).mean()*100:.2f} | v4.1 {np.abs(v-y).mean()*100:.2f} | Eagle GNN {np.abs(eg-y).mean()*100:.2f}")
        print(f"  Heron-trained GNN per-seed MAE {[round(r,2) for r in res]} mean {np.mean(res):.2f}; ensemble MAE {np.abs(pm-y).mean()*100:.2f}, rank corr {np.corrcoef(np.argsort(np.argsort(pm)), np.argsort(np.argsort(y)))[0,1]:.2f}, bias {(pm-y).mean()*100:+.2f} | {time.time()-t0:.0f}s")
        json.dump({"y": [float(x) for x in y], "v41": [float(x) for x in v], "gnn_ensemble": [float(x) for x in pm], "seeds_mae": [float(r) for r in res]}, open("quantumbridge_data/entry112_hw_results.json", "w"))
