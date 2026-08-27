"""
QuantumBridge — Entry 049b: collect real-hardware results and compare
against v4.1 / GNN / Aer-simulated predictions.

Run this after entry049_submit_real_jobs.py's jobs have finished (check
status at https://quantum.ibm.com under your account's job history, or
this script will just tell you if a job isn't done yet). Also needs
IBM_QUANTUM_TOKEN set the same way as the submit script.

Produces quantumbridge_data/entry049_real_results.json and prints a
comparison table: v4.1 vs GNN vs Aer-simulated vs REAL HARDWARE, which is
the number this whole project has been missing until now.
"""

import json
import os

from qiskit_ibm_runtime import QiskitRuntimeService

JOB_IDS_PATH = "quantumbridge_data/entry049_job_ids.json"
OUT_PATH = "quantumbridge_data/entry049_real_results.json"


def ideal_bitstrings(n):
    return {"0" * n, "1" * n}


def main():
    token = os.environ.get("IBM_QUANTUM_TOKEN")
    if not token:
        print("ERROR: set the IBM_QUANTUM_TOKEN environment variable first.")
        return

    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token,
                                   instance="open-instance")
    job_ids = json.load(open(JOB_IDS_PATH))

    all_results = []
    for chip, info in job_ids.items():
        job = service.job(info["job_id"])
        status = job.status()
        print(f"{chip} ({info['backend']}): job {info['job_id']} status = {status}")
        if status != "DONE":
            print("  not finished yet -- rerun this script later")
            continue

        result = job.result()
        for i, cand in enumerate(info["candidates"]):
            pub_result = result[i]
            counts = pub_result.data.meas.get_counts() if hasattr(pub_result.data, "meas") \
                else pub_result.data.c.get_counts()
            n_meas = len(cand["meas_qubits"])
            ideal = ideal_bitstrings(n_meas)
            total = sum(counts.values())
            ok = sum(c for bitstr, c in counts.items() if bitstr.replace(" ", "") in ideal)
            real_success = ok / total

            row = {**cand, "real_hardware_result": real_success}
            all_results.append(row)
            gap_v41 = abs(row["v4_1"] - real_success)
            gap_aer = abs(row["aer_sim"] - real_success)
            print(f"  [{cand['tag']}] {chip} {cand.get('a', cand.get('qubits'))}: "
                 f"v4.1={row['v4_1']*100:.1f}% aer_sim={row['aer_sim']*100:.1f}% "
                 f"REAL={real_success*100:.1f}%  "
                 f"(gap vs v4.1: {gap_v41*100:.1f}pt, gap vs aer_sim: {gap_aer*100:.1f}pt)")

    if all_results:
        json.dump(all_results, open(OUT_PATH, "w"), indent=2)
        print(f"\nsaved {len(all_results)} real-hardware comparisons -> {OUT_PATH}")

        import statistics
        gaps_v41 = [abs(r["v4_1"] - r["real_hardware_result"]) for r in all_results]
        gaps_aer = [abs(r["aer_sim"] - r["real_hardware_result"]) for r in all_results]
        print(f"\nMean |v4.1 - real|:     {statistics.mean(gaps_v41)*100:.2f} pts")
        print(f"Mean |aer_sim - real|:  {statistics.mean(gaps_aer)*100:.2f} pts")
        print("(if aer_sim tracks real hardware closely, the simulator -- and everything")
        print(" trained against it -- is a reasonable proxy; if not, that's the real finding.)")
    else:
        print("\nno completed jobs yet.")


if __name__ == "__main__":
    main()
