from emulator_v3_routing import load_calibration, build_connectivity_graph, cnot_error_for_pair, success_prob_from_gate_count

calibration = load_calibration()
graph = build_connectivity_graph(calibration)

err, notes = cnot_error_for_pair(graph, 0, 26)
print(f"cnot_error_for_pair (path-aware, forgiveness-corrected): {(1-err)*100:.2f}% success")
print(f"  {notes[0]}")

avg_based = success_prob_from_gate_count(34, graph)
print(f"\nsuccess_prob_from_gate_count (avg-based, 34 real CX): {avg_based*100:.2f}% success")

print(f"\nAer real noise result (Entry 017): 95.19% success")
