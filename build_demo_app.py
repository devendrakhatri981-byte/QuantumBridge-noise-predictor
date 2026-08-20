"""Build the self-contained QuantumBridge public demo HTML file, embedding
the precomputed 16,002-pair lookup table (both chips) directly so it works
fully offline with no server or Qiskit/JAX dependency."""

import json

lut = json.load(open("quantumbridge_data/entry046_lookup.json"))
lut_json = json.dumps(lut, separators=(",", ":"))

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>QuantumBridge — Live Noise Prediction Demo</title>
<style>
  :root {
    --bg: #0b1220; --panel: #121b2e; --panel2: #17233a; --text: #e8ecf5;
    --muted: #8a96b5; --accent: #F0997B; --accent2: #7F77DD; --good: #6fcf97;
    --bad: #eb5757; --border: #253150;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    padding: 28px 18px 60px;
  }
  .wrap { max-width: 880px; margin: 0 auto; }
  h1 { font-size: 22px; margin: 0 0 4px; }
  .sub { color: var(--muted); font-size: 13.5px; margin-bottom: 20px; line-height: 1.5; }
  .card {
    background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
    padding: 20px; margin-bottom: 16px;
  }
  .row { display: flex; gap: 14px; flex-wrap: wrap; align-items: flex-end; }
  label { display: block; font-size: 12px; color: var(--muted); margin-bottom: 5px; }
  select, input[type=number] {
    background: var(--panel2); border: 1px solid var(--border); color: var(--text);
    padding: 9px 10px; border-radius: 8px; font-size: 14px; width: 130px;
  }
  button {
    background: var(--accent); color: #1a1200; border: none; border-radius: 8px;
    padding: 10px 18px; font-size: 14px; font-weight: 600; cursor: pointer;
  }
  button.secondary { background: var(--panel2); color: var(--text); border: 1px solid var(--border); }
  button:hover { opacity: 0.9; }
  .results { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px,1fr)); gap: 12px; margin-top: 18px; }
  .metric { background: var(--panel2); border-radius: 10px; padding: 14px; text-align: center; }
  .metric .label { font-size: 11.5px; color: var(--muted); margin-bottom: 6px; }
  .metric .val { font-size: 22px; font-weight: 700; }
  .metric.gnn .val { color: var(--accent); }
  .metric.v41 .val { color: var(--accent2); }
  .metric.real .val { color: var(--good); }
  .metric.real.missing .val { color: var(--muted); font-size: 14px; }
  .note { font-size: 12px; color: var(--muted); margin-top: 14px; line-height: 1.6; }
  .badge {
    display: inline-block; font-size: 11px; padding: 2px 8px; border-radius: 999px;
    background: var(--panel2); color: var(--muted); border: 1px solid var(--border);
    margin-left: 6px;
  }
  .badge.floor { background: #3a1f1f; color: #eb9a9a; border-color: #5a2b2b; }
  .footer { color: var(--muted); font-size: 11.5px; margin-top: 24px; line-height: 1.7; }
  a { color: var(--accent); }
  .stats-strip { display: flex; gap: 22px; flex-wrap: wrap; font-size: 12px; color: var(--muted); margin-top: 6px; }
  .stats-strip b { color: var(--text); }
</style>
</head>
<body>
<div class="wrap">
  <h1>QuantumBridge — Live Circuit Reliability Predictor</h1>
  <div class="sub">
    Pick a real IBM chip layout and any two qubits, and get a Bell-pair circuit's predicted
    execution success from two models: <b style="color:var(--accent2)">v4.1</b>, a hand-derived
    closed-form physics model, and <b style="color:var(--accent)">GNN</b>, a graph neural network
    trained on 2,317 real noise-simulated circuits. Where available, the actual simulated hardware
    result is shown alongside for comparison.
    <div class="stats-strip">
      <span><b>16,002</b> qubit pairs covered</span>
      <span><b>2,317</b> circuits in training data</span>
      <span><b>127</b> qubits per chip (Kyiv &amp; Sherbrooke)</span>
      <span><b>2,737</b> GNN parameters</span>
    </div>
  </div>

  <div class="card">
    <div class="row">
      <div>
        <label>Chip</label>
        <select id="chip">
          <option value="kyiv">IBM Kyiv (127q)</option>
          <option value="sherbrooke" selected>IBM Sherbrooke (127q)</option>
        </select>
      </div>
      <div>
        <label>Qubit A</label>
        <input type="number" id="qa" min="0" max="126" value="3">
      </div>
      <div>
        <label>Qubit B</label>
        <input type="number" id="qb" min="0" max="126" value="6">
      </div>
      <button onclick="predict()">Predict</button>
      <button class="secondary" onclick="randomPair()">Random pair</button>
      <button class="secondary" onclick="floorExample()">Show floor-collapse example</button>
    </div>

    <div class="results" id="results" style="display:none;">
      <div class="metric">
        <div class="label">Real routed distance</div>
        <div class="val" id="r-hop">—</div>
      </div>
      <div class="metric v41">
        <div class="label">v4.1 closed-form model</div>
        <div class="val" id="r-v41">—</div>
      </div>
      <div class="metric gnn">
        <div class="label">GNN prediction</div>
        <div class="val" id="r-gnn">—</div>
      </div>
      <div class="metric real" id="r-real-box">
        <div class="label">Simulated hardware result</div>
        <div class="val" id="r-real">—</div>
      </div>
    </div>
    <div class="note" id="note"></div>
  </div>

  <div class="footer">
    <b>What this is, honestly:</b> every prediction shown is a real output from the actual trained
    model and the actual closed-form formula — nothing here is faked or approximated for the demo.
    But the underlying noise data comes from Qiskit's "fake backend" snapshots of real IBM device
    calibration, run through Aer's noise simulator — not live queued jobs on physical hardware.
    Ground truth is only available for the subset of pairs that were actually simulated
    (about 1,600 of the 16,002 shown here); the rest show model predictions only, clearly marked.
    This is an early-stage academic project (undergraduate research), not a production tool.
    Built with Qiskit, JAX/Optax, and a from-scratch graph neural network.
  </div>
</div>

<script>
const LUT = __LUT_JSON__;

function findRow(chip, a, b) {
  const lo = Math.min(a,b), hi = Math.max(a,b);
  const rows = LUT[chip];
  // rows are pre-sorted by (a,b) since generated a<b in order
  for (const r of rows) {
    if (r[0] === lo && r[1] === hi) return r;
  }
  return null;
}

function render(row) {
  document.getElementById('results').style.display = 'grid';
  const [a,b,hop,nn,ne,v41,gnn,real] = row;
  document.getElementById('r-hop').textContent = hop + ' hop' + (hop===1?'':'s') + ' (' + nn + ' nodes)';
  document.getElementById('r-v41').textContent = (v41*100).toFixed(1) + '%';
  document.getElementById('r-gnn').textContent = (gnn*100).toFixed(1) + '%';
  const realBox = document.getElementById('r-real-box');
  const realEl = document.getElementById('r-real');
  if (real === null || real === undefined) {
    realBox.classList.add('missing');
    realEl.textContent = 'not simulated';
  } else {
    realBox.classList.remove('missing');
    realEl.textContent = (real*100).toFixed(1) + '%';
  }
  let note = '';
  if (real !== null && real !== undefined && Math.abs(real-0.5) < 0.03 && hop <= 10) {
    note += '<span class="badge floor">floor-collapse case</span> This circuit\\'s real success ' +
      'crashed to the 50% measurement floor despite a short route -- exactly the failure mode ' +
      'the closed-form v4.1 model structurally cannot see, and the GNN was built to catch. ';
  }
  note += 'Route length and prediction shown are from the real Qiskit-transpiled circuit for this ' +
    'exact pair on this chip -- not an approximation.';
  document.getElementById('note').innerHTML = note;
}

function predict() {
  const chip = document.getElementById('chip').value;
  let a = parseInt(document.getElementById('qa').value);
  let b = parseInt(document.getElementById('qb').value);
  if (isNaN(a) || isNaN(b) || a === b || a < 0 || b < 0 || a > 126 || b > 126) {
    alert('Pick two different qubits between 0 and 126.'); return;
  }
  const row = findRow(chip, a, b);
  if (!row) { alert('No data for that pair (unexpected -- all pairs should be covered).'); return; }
  render(row);
}

function randomPair() {
  const chip = document.getElementById('chip').value;
  const rows = LUT[chip];
  const row = rows[Math.floor(Math.random()*rows.length)];
  document.getElementById('qa').value = row[0];
  document.getElementById('qb').value = row[1];
  render(row);
}

function floorExample() {
  document.getElementById('chip').value = 'sherbrooke';
  document.getElementById('qa').value = 3;
  document.getElementById('qb').value = 6;
  predict();
}

// show the floor-collapse example on load
floorExample();
</script>
</body>
</html>
"""

html = HTML.replace("__LUT_JSON__", lut_json)
with open("quantumbridge_live_demo.html", "w") as f:
    f.write(html)

import os
print("saved quantumbridge_live_demo.html,", os.path.getsize("quantumbridge_live_demo.html")/1e6, "MB")
