# QuantumBridge Live Inference API

A real backend for the QuantumBridge noise predictor, replacing the static
demo's precomputed 24,003-row lookup table with live inference: every
request runs an actual Qiskit SABRE-routed transpile, the v4.1 closed-form
physics prediction, and an Entry 071 graph neural network forward pass
(with calibrated MC-Dropout uncertainty), computed on demand.

This is a deliberately scoped v1, not the full "any circuit, any chip"
vision:

- **Circuit type:** Bell pairs only (H + CX between two qubits) -- what
  the model was trained and validated on.
- **Chips:** `kyiv`, `sherbrooke`, `brisbane` -- the three chips the
  deployed model (`entry071_deploy_params.json`) was trained on. A fourth
  chip is refused with an explicit message rather than silently
  extrapolated, since Entry 073's true zero-shot test found the model's
  accuracy edge over the simpler v4.1 formula doesn't clearly survive a
  genuinely unseen chip.
- **Validation:** noise data comes from Qiskit's fake-backend snapshots
  run through Aer's simulator, not live queued jobs on real IBM hardware.
  This is stated in every `/health` and `/predict` response, not just here.

## Requirements

This API depends on modules and data files from the main QuantumBridge
project (`emulator_v3_routing.py`, `emulator_v4.py`,
`exact_dwell_routing.py`, `entry044_build_graphs.py`, and everything under
`quantumbridge_data/`). Keep this `api/` folder inside the project
repository -- it is not a standalone package.

```
QuantumBridge_backup/
  api/
    main.py
    requirements.txt
    README.md          <- this file
  emulator_v3_routing.py
  emulator_v4.py
  exact_dwell_routing.py
  entry044_build_graphs.py
  quantumbridge_data/
    entry071_deploy_params.json
    offline_calibration_{kyiv,sherbrooke,brisbane}_full.json
    real_topology_{kyiv,sherbrooke,brisbane}.json
    coherence_{kyiv,sherbrooke,brisbane}.json
```

## Run locally

```bash
cd QuantumBridge_backup
pip install -r api/requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Then:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/chips
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"chip": "kyiv", "qubit_a": 0, "qubit_b": 11}'
```

Interactive API docs (Swagger UI) are auto-generated at
`http://localhost:8000/docs`.

## Endpoints

**`GET /health`**
Returns supported chips, circuit types, model info, and the validation
caveat.

**`GET /chips`**
Returns qubit counts for each supported chip.

**`POST /predict`**
Body: `{"chip": "kyiv" | "sherbrooke" | "brisbane", "qubit_a": int, "qubit_b": int}`

Returns:
```json
{
  "chip": "kyiv",
  "qubit_a": 0,
  "qubit_b": 11,
  "bfs_hop_distance": 11,
  "n_nodes": 12,
  "n_edges": 11,
  "v4_1_prediction": 0.762,
  "gnn_prediction": 0.7981,
  "gnn_uncertainty": 0.0053,
  "note": "...",
  "latency_ms": 18.4
}
```

`gnn_prediction` is the mean of 20 stochastic MC-Dropout forward passes;
`gnn_uncertainty` is their standard deviation. This will vary slightly
(typically < 1 percentage point) between identical requests since dropout
is genuinely stochastic -- that variance is itself part of what the
uncertainty estimate is measuring, not a bug.

## Wiring up the live demo

`quantumbridge_live_demo.html` already knows how to call this API -- once
deployed, open the file and set the `API_BASE_URL` constant near the top
of the `<script>` block:

```js
const API_BASE_URL = "https://your-app.onrender.com";
```

With this set, the demo tries a live request first (4s timeout) and shows
a green "live prediction" badge; if the API is unreachable or
`API_BASE_URL` is left empty, it falls back to the precomputed table with
a "cached / precomputed" badge. The demo works correctly either way --
this is a progressive upgrade, not a hard dependency. Don't forget CORS is
already enabled (`allow_origins=["*"]` in `main.py`) so the browser fetch
from GitHub Pages (or wherever the demo is hosted) will succeed.

Unsupported chips, equal qubit indices, and out-of-range qubit indices
all return `400` with a specific message rather than a generic error.

## Deployment

This project was built and validated in a local/sandboxed environment;
Claude cannot create hosting accounts or push deployments on your behalf.
To put this online yourself, any platform that runs a standard Python web
service works -- for example:

- **Render** or **Railway**: connect your GitHub repo, set the start
  command to `uvicorn api.main:app --host 0.0.0.0 --port $PORT`, and set
  the build command to `pip install -r api/requirements.txt`.
- **Fly.io**: similar, via a `Dockerfile` or their Python buildpack.
- **A VPS you control**: `pip install -r api/requirements.txt` then run
  uvicorn behind a reverse proxy (nginx/Caddy) with TLS.

Whichever you pick, budget for cold-start latency: the model and per-chip
calibration data load once at process startup (a few seconds), then every
request after that is fast (the local test above showed 18-110ms per
prediction).

## What this does NOT do (yet)

- No GHZ, star, or chain circuit support in the live API, even though the
  training data includes them -- only the bell-pair feature-extraction
  path has been wired and tested here.
- No real IBM hardware validation -- everything is fake-backend + Aer
  simulation, disclosed in every response.
- No authentication, rate limiting, or request logging -- add these
  before exposing this publicly at any real traffic volume.
- No support for a chip outside the three trained ones. See Entry 073 in
  the research log before considering whether to add a new chip without
  retraining.
